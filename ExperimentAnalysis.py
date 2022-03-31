from importlib.metadata import distribution
from typing import Union, Dict, Callable, List
import persistence
from utils import compute_time_interval, restrict_to_interval
from analysis import (
    detect_peaks,
    instantaneous_rate_from_spike_train,
    multitaper_power_spectral_density,
    restrict_frequency,
    gaussian_smoothing,
    snr,
    cell_rate_from_spike_train,
    effective_total_synaptic_conductance,
    synchronization_frequency,
)
import attractor
import numpy as np
import copy
import parse_equations


class ExperimentAnalysis:
    """
    Analyse data from :class:`BrianExperiment.BrianExperiment`

    Example for analyzing data by all analysis methods

    .. testsetup::

        import numpy as np
        from BrianExperiment import BrianExperiment
        from  ExperimentAnalysis import ExperimentAnalysis
        from persistence import FileMap
        from utils import TestEnv

    .. testcode::

        with TestEnv():
            for run in range(2):
                with BrianExperiment(persist=True, path="file.h5", object_path=f"/run_{run}/data") as exp:
                    exp.persist_data["mon"] = np.arange(10)
            with FileMap("file_analysis.h5") as af:
                with FileMap("file.h5") as f:
                    for run in f.keys():
                        exp_analysis = ExperimentAnalysis(experiment_data=f[run]["data"])
                        exp_analysis.analyze_all()
                        af[run] = exp_analysis.report()

                print({run:list(af[run].keys()) for run in af.keys()})


    .. testoutput::

        { "run_1" : ["x", "y", "z"], "run_2" : ["x", "y", "z"] }


    Example for analyzing data by specific analysis methods

    .. testsetup::

        import numpy as np
        from BrianExperiment import BrianExperiment
        from  ExperimentAnalysis import ExperimentAnalysis
        from persistence import FileMap
        from utils import TestEnv

    .. testcode::

        with TestEnv():
            for run in range(2):
                with BrianExperiment(persist=True, path="file.h5", object_path=f"/run_{run}/data") as exp:
                    exp.persist_data["mon"] = np.arange(10)
            with FileMap("file_analysis.h5") as af:
                with FileMap("file.h5") as f:
                    for run in f.keys():
                        exp_analysis = ExperimentAnalysis(experiment_data=f[run]["data"])
                        exp_analysis.analyze_instantaneous_rate()
                        af[run] = exp_analysis.report()

                print({run:list(af[run].keys()) for run in af.keys()})


    .. testoutput::

        { "run_1" : ["instantaneous_rate"], "run_2" : ["instantaneous_rate"] }
    """

    def __init__(
        self,
        experiment_data: Union[Dict, persistence.Reader],
        t_start: float = 10.0,
        t_end: float = None,
    ):
        """

        :param t_start: if set consider only data produced from (incl.) this time point
        :param t_end: if set consider only data produced until (incl.) this time point

        all analyses will be done  only on data within the interval [t_start, t_end]
        """

        self._data = experiment_data
        self._analysis = {
            cat: {instance: {} for instance in self._data[cat].keys()}
            for cat in self._data.keys()
        }

        self.dt = experiment_data["meta"]["dt"]["value"] * 1000
        t = experiment_data["meta"]["t"]["value"] * 1000

        self.t_start, self.t_end, _ = compute_time_interval(t, self.dt, t_start, t_end)
        self._analysis["meta"] = {
            "t_start": {"value": self.t_start / 1000, "unit": "s"},
            "t_end": {"value": self.t_end / 1000, "unit": "s"},
            "dt": {"value": self.dt / 1000, "unit": "s"},
        }

    @property
    def report(self):
        return self._analysis

    def analyze_all(self):
        analysis_functions = [
            getattr(self, f_name)
            for f_name in dir(self)
            if isinstance(getattr(self, f_name), Callable)
            and f_name.startswith("analyze_")
            and f_name != "analyze_all"
        ]
        for f in analysis_functions:
            f()

    def analyze_instantaneous_rate(self, pop_name: List[str] = None):
        """
        compute the instantaneous rate for populations

        :param pop_name: populations for which the instantaneous rate is computed
                         - df: None ~ all populations
        """
        pop_name = (
            list(self._data["SpikeDeviceGroup"].keys())
            if pop_name == None
            else pop_name
        )
        for pn in pop_name:
            # print(pn, self._data["SpikeDeviceGroup"][pn].keys())
            if (
                self._data["SpikeDeviceGroup"][pn]["device"]["class"][0]
                == "NeuronPopulation"
            ):

                spike_train = {
                    neuron: train * 1000
                    for neuron, train in self._data["SpikeDeviceGroup"][pn]["spike"][
                        "spike_train"
                    ]["value"].items()
                }
                inst_rate = instantaneous_rate_from_spike_train(
                    self._data["meta"]["t"]["value"] * 1000,
                    self._data["meta"]["dt"]["value"] * 1000,
                    spike_train,
                )

                inst_rate, _, _ = restrict_to_interval(
                    inst_rate, self.dt, self.t_start, self.t_end
                )

                self._analysis["SpikeDeviceGroup"][pn]["instantaneous_rate"] = {
                    "value": inst_rate,
                    "unit": "Hz",
                }

    def analyze_smoothed_rate(self, mode: str = "gaussian", window_size: float = 1.0):
        """
        compute the smoothed rate for populations analyzed in :meth:`ExperimentAnalysis.analyze_instantaneous_rate`

        :param mode: mode used for smoothing the instantaneous rate
        :param window_size: window size tb used for smoothing in [ms]
        """
        if mode not in ["gaussian"]:
            raise ValueError(f"param mode must be one of ['gaussian']. Is {mode}")

        for pn in self._analysis["SpikeDeviceGroup"].keys():
            if (
                not "smoothed_rate" in self._analysis["SpikeDeviceGroup"][pn].keys()
                and self._data["SpikeDeviceGroup"][pn]["device"]["class"][0]
                == "NeuronPopulation"
                and "instantaneous_rate"
                in self._analysis["SpikeDeviceGroup"][pn].keys()
            ):
                if mode == "gaussian":
                    val = gaussian_smoothing(
                        self._analysis["SpikeDeviceGroup"][pn]["instantaneous_rate"][
                            "value"
                        ],
                        window_size=4.0,
                        one_sigma_window=1.0,
                        dt=self._data["meta"]["dt"]["value"] * 1000,
                    )
                # furhter windows elif ....

                # note that all data was produced in [t_start, t_end] as this measure depends only on instantaneous_rate
                self._analysis["SpikeDeviceGroup"][pn]["smoothed_rate"] = {
                    "value": val,
                    "mode": mode,
                    "unit": "Hz",
                }

    def analyze_power_spectral_density(
        self,
        pop_name: List[str] = None,
        separate_intervals: bool = False,
        f_lower_bound: float = 50.0,
        f_upper_bound: float = 300.0,
    ):
        """
        compute power spectral density
        - computes psd over the entire signal if separate_intervals set also computes psd in separate time intervals

        :param pop_name: list of populations to compute psd for - defaults to all NeuronPopulations
        :param separate_intervals: also compute psd for separate time intervals (using sliding window)
        :param f_lower_bound: does not consider frequencies (and resp. psd) below lower bound
        :param f_upper_bound: does not consider frequencies (and resp. psd) above upper bound
        """
        # time resolution of instantaneous rate is simulation time step as it is manually computed from spike trains
        dt = self.dt

        pop_name = (
            list(self._data["SpikeDeviceGroup"].keys())
            if pop_name == None
            else pop_name
        )

        for pn in pop_name:
            if (
                self._data["SpikeDeviceGroup"][pn]["device"]["class"][0]
                == "NeuronPopulation"
            ):
                # note that all data was produced in [t_start, t_end] as this measure depends only on instantaneous_rate
                # t_start and t_end are now set such that inst_rate contains all samples within [t_start, t_end] (closed bounds)
                inst_rate = self._analysis["SpikeDeviceGroup"][pn][
                    "instantaneous_rate"
                ]["value"]


                freq, psd = multitaper_power_spectral_density(inst_rate, dt)

                freq, psd = restrict_frequency(
                    freq, psd, f_lower_bound=f_lower_bound, f_upper_bound=f_upper_bound
                )

                self._analysis["SpikeDeviceGroup"][pn]["psd_complete"] = {
                    "psd": psd,
                    "frequency": {"value": freq, "unit": "Hz"},
                }

                if separate_intervals:
                    w_sliding = 2 ** 13
                    # setting nfft in this manner increases resolution and leads to minor speed up even though
                    # (resolution/ # discrete points computed is larger)
                    freq, psd = multitaper_power_spectral_density(
                        inst_rate, dt, w_sliding=w_sliding, nfft=w_sliding * 2
                    )
                    freq, psd = restrict_frequency(
                        freq,
                        psd,
                        f_lower_bound=f_lower_bound,
                        f_upper_bound=f_upper_bound,
                    )

                    self._analysis["SpikeDeviceGroup"][pn]["psd_interval"] = {
                        "psd": psd,
                        "frequency": {"value": freq, "unit": "Hz"},
                    }

    def analyze_peaks(self, pop_name: List[str] = None, smoothed: bool = True):
        """
        analyze the peaks (& troughs) of  the population rate of neuron populations
        with a minimum distance between peaks of half the wavelength of the fundamental
        frequency of the population rate

        :param pop_name: population names for which peaks are tb detected
        :param smoothed: whether to analyze the smoothed or instantaneous population rate
        """

        pop_name = (
            list(self._data["SpikeDeviceGroup"].keys())
            if pop_name == None
            else pop_name
        )

        for pn in pop_name:
            if (
                self._data["SpikeDeviceGroup"][pn]["device"]["class"][0]
                == "NeuronPopulation"
                and pn in self._analysis["SpikeDeviceGroup"]
            ):
                if smoothed:
                    pop_rate = self._analysis["SpikeDeviceGroup"][pn]["smoothed_rate"][
                        "value"
                    ]
                else:
                    pop_rate = self._analysis["SpikeDeviceGroup"][pn][
                        "instantaneous_rate"
                    ]["value"]

                mins, maxs = detect_peaks(-pop_rate, self.dt), detect_peaks(
                    pop_rate, self.dt
                )

                self._analysis["SpikeDeviceGroup"][pn]["peaks"] = {
                    "mins": mins,
                    "maxs": maxs,
                    "smoothed": smoothed,
                }

    def analyze_cell_rate(self, pop_name: List[str] = None):
        """
        cell rate per cell, the population average and the population maximum for populations

        :param pop_name: populations for which the instantaneous rate is computed
                         - df: None ~ all populations
        """
        pop_name = (
            list(self._data["SpikeDeviceGroup"].keys())
            if pop_name == None
            else pop_name
        )
        for pn in pop_name:
            if (
                self._data["SpikeDeviceGroup"][pn]["device"]["class"][0]
                == "NeuronPopulation"
            ):

                spike_train = {
                    neuron: train * 1000
                    for neuron, train in self._data["SpikeDeviceGroup"][pn]["spike"][
                        "spike_train"
                    ]["value"].items()
                }

                ids = self._data["SpikeDeviceGroup"][pn]["ids"]

                ids, cell_rate = cell_rate_from_spike_train(
                    self.t_start,
                    self.t_end,
                    ids,
                    spike_train,
                )

                self._analysis["SpikeDeviceGroup"][pn]["cell_rate"] = {
                    "ids": ids,
                    "cell_rate": {"value": cell_rate, "unit": "Hz"},
                    "pop_avg_cell_rate": {"value": np.mean(cell_rate), "unit": "Hz"},
                    "pop_max_cell_rate": {"value": np.max(cell_rate), "unit": "Hz"},
                }

    def analyze_snr(self, bin_size=10.0):
        """
        signal-to-noise ratio for populations analyzed in :meth:`ExperimentAnalysis.analyze_power_spectral_density`
        """
        for pn in self._analysis["SpikeDeviceGroup"].keys():
            if (
                "psd_complete" in self._analysis["SpikeDeviceGroup"][pn].keys()
                and self._data["SpikeDeviceGroup"][pn]["device"]["class"][0]
                == "NeuronPopulation"
            ):
                psd = self._analysis["SpikeDeviceGroup"][pn]["psd_complete"]["psd"]
                freq = self._analysis["SpikeDeviceGroup"][pn]["psd_complete"][
                    "frequency"
                ]["value"]

                ratio = snr(psd, freq, bin_size)

                self._analysis["SpikeDeviceGroup"][pn]["snr"] = {
                    "snr": ratio,
                    "bin_size": bin_size,
                }

    def analyze_synchronization_frequency(self):
        """
        synchronization_frequency for populations analyzed in :meth:`ExperimentAnalysis.analyze_power_spectral_density`
        """
        for pn in self._analysis["SpikeDeviceGroup"].keys():
            if (
                "psd_complete" in self._analysis["SpikeDeviceGroup"][pn].keys()
                and self._data["SpikeDeviceGroup"][pn]["device"]["class"][0]
                == "NeuronPopulation"
            ):
                psd = self._analysis["SpikeDeviceGroup"][pn]["psd_complete"]["psd"]
                freq = self._analysis["SpikeDeviceGroup"][pn]["psd_complete"][
                    "frequency"
                ]["value"]

                f, p = synchronization_frequency(freq, psd)

                self._analysis["SpikeDeviceGroup"][pn]["synchronization_frequency"] = {
                    "frequency": {"value": f, "unit": "Hz"},
                    "power": p,
                }

    def analyze_avg_power(self):
        """
        average power for populations analyzed in :meth:`ExperimentAnalysis.analyze_power_spectral_density`
        """
        for pn in self._analysis["SpikeDeviceGroup"].keys():
            if (
                "psd_complete" in self._analysis["SpikeDeviceGroup"][pn].keys()
                and self._data["SpikeDeviceGroup"][pn]["device"]["class"][0]
                == "NeuronPopulation"
            ):
                psd = self._analysis["SpikeDeviceGroup"][pn]["psd_complete"]["psd"]

                avg_pwr = np.mean(psd)

                self._analysis["SpikeDeviceGroup"][pn]["avg_power"] = avg_pwr

    def analyze_total_synaptic_conductance(
        self,
        pop_e: str,
        pop_i: str,
        synaptic_input_e_e_name: str = "x_AMPA",
        synaptic_input_i_e_name: str = "x_GABA",
        conductance_e_e_name: str = "gsynE_E",
        conductance_i_e_name: str = "gsynI_E",
    ):
        """
        total synaptic conductance for populations pop_e assumes synaptic connectivity to pop_e
        is limited to e-e: pop_e -> pop_e and i-e: pop_i -> pop_e

        :param pop_e: excitatory population for which total synaptic conductance is computed
        :param pop_i: inhibitory population which connects to pop_e
        :param synaptic_input_e_e_name: name of synaptic input variable that is modified on presynaptic spike for e-e synapses
        :param synaptic_input_i_e_name: name of synaptic input variable that is modified on presynaptic spike for i-e synapses
        :param conductance_e_e_name: name of conductance for synaptic inputs of e-e synapses
        :param conductance_i_e_name: name of conductance for synaptic inputs of i-e synapses
        :return: total conductance and respective ids of pop_e
        """
        synapse_e_e = [
            k
            for k, v in self._data["Synapse"].items()
            if v["source"]["name"] == pop_e and v["target"]["name"] == pop_e
        ]
        synapse_i_e = [
            k
            for k, v in self._data["Synapse"].items()
            if v["source"]["name"] == pop_i and v["target"]["name"] == pop_e
        ]
        if not (
            pop_e in self._data["SpikeDeviceGroup"].keys()
            and pop_i in self._data["SpikeDeviceGroup"].keys()
        ):
            raise ValueError("No such populations pop_e or pop_i.")
        if not (
            "cell_rate" in self._analysis["SpikeDeviceGroup"][pop_e]
            and "cell_rate" in self._analysis["SpikeDeviceGroup"][pop_i]
        ):
            raise ValueError(
                f"cell rate must be analyzed for pop {pop_e} and pop {pop_i} prior to calling this method"
            )
        if not (len(synapse_e_e) > 0 and len(synapse_i_e) > 0):
            raise ValueError(
                f"No synapses matching {pop_e} or {pop_i} found - both synapses E->E and E->I must be present."
            )
        else:
            synapse_e_e = synapse_e_e[0]
            synapse_i_e = synapse_i_e[0]

        if (
            "on_pre" not in self._data["Synapse"][synapse_e_e]["synapse_params"].keys()
            or "on_pre"
            not in self._data["Synapse"][synapse_i_e]["synapse_params"].keys()
        ):
            raise ValueError(
                "on_pre not in synapse_params for {synapse_e_e}, {synapse_i_e} - "
                + "total conductance cannot be analyzed without on_pre being set on the respective synapses for simulation."
            )

        source_ids_e_e = self._data["Synapse"][synapse_e_e]["source"]["ids"]
        target_ids_e_e = self._data["Synapse"][synapse_e_e]["target"]["ids"]

        source_ids_i_e = self._data["Synapse"][synapse_i_e]["source"]["ids"]
        target_ids_i_e = self._data["Synapse"][synapse_i_e]["target"]["ids"]

        cell_rate_e = self._analysis["SpikeDeviceGroup"][pop_e]["cell_rate"][
            "cell_rate"
        ]["value"]
        cell_rate_i = self._analysis["SpikeDeviceGroup"][pop_i]["cell_rate"][
            "cell_rate"
        ]["value"]

        # parameters
        parameters = self._data["meta"]["neuron_parameters"].load()

        # parameters e_e
        conductance_const_e_e = parameters[conductance_e_e_name]["value"]

        synapse_params_e_e = self._data["Synapse"][synapse_e_e]["synapse_params"].load()
        on_pre_e_e = self._data["Synapse"][synapse_e_e]["synapse_params"]["on_pre"][0]

        neuron_variables_e_e = parse_equations.extract_variables_from_equations(
            on_pre_e_e
        )

        for k, v in neuron_variables_e_e.items():
            synapse_params_e_e[k] = v["neutral_element"]

        parameters_e_e = copy.deepcopy(parameters)
        parameters_e_e.update(synapse_params_e_e)

        # parameters i_e
        conductance_const_i_e = parameters[conductance_i_e_name]["value"]

        synapse_params_i_e = self._data["Synapse"][synapse_i_e]["synapse_params"].load()
        on_pre_i_e = self._data["Synapse"][synapse_i_e]["synapse_params"]["on_pre"][0]

        neuron_variables_i_e = parse_equations.extract_variables_from_equations(
            on_pre_i_e
        )

        for k, v in neuron_variables_i_e.items():
            synapse_params_i_e[k] = v["neutral_element"]

        parameters_i_e = parameters
        parameters_i_e.update(synapse_params_i_e)

        # syn_const e_e
        eval_on_pre_e_e = parse_equations.evaluate_equations(on_pre_e_e, parameters_e_e)

        syn_const_e_e = eval_on_pre_e_e[synaptic_input_e_e_name]

        # syn_const i_e
        eval_on_pre_i_e = parse_equations.evaluate_equations(on_pre_i_e, parameters_i_e)

        syn_const_i_e = eval_on_pre_i_e[synaptic_input_i_e_name]

        eff_total_conductance = effective_total_synaptic_conductance(
            source_ids_e_e,
            target_ids_e_e,
            source_ids_i_e,
            target_ids_i_e,
            cell_rate_e,
            cell_rate_i,
            syn_const_e_e,
            syn_const_i_e,
            conductance_const_e_e,
            conductance_const_i_e,
        )

        self._analysis["SpikeDeviceGroup"][pop_e][
            "total_synaptic_conductance"
        ] = eff_total_conductance

    def analyze_snapshots(self, pop_name: str):
        """
        extract snapshots from population rate and spike trains

        :param pop_name: population name for which to extract snapshots
        """
        if not pop_name in self._data["SpikeDeviceGroup"].keys():
            raise ValueError("No such population {pop_name}.")
        if not "smoothed_rate" in self._analysis["SpikeDeviceGroup"][pop_name].keys():
            raise ValueError("'smoothed_rate' must be analyzed first.")

        rate = self._analysis["SpikeDeviceGroup"][pop_name]["smoothed_rate"]["value"]
        spike_train = {
            neuron: train * 1000
            for neuron, train in self._data["SpikeDeviceGroup"][pop_name]["spike"][
                "spike_train"
            ]["value"].items()
        }
        ids = self._data["SpikeDeviceGroup"][pop_name]["ids"]
        dt = self.dt

        t_start = self.t_start
        t_end = self.t_end

        self._analysis["SpikeDeviceGroup"][pop_name][
            "snapshots"
        ] = attractor.extract_snapshots(spike_train, ids.size, rate, t_start, t_end, dt)

    def analyze_similarity_distribution(self, pop_name: str):
        """
        compute similarity distribution across snapshots per pattern
        (distribution: (N x C), (i,j) ~ similarity of snapshot from cycle j with pattern i)

        assumption: patterns used for pop_name are assigned to the :attr:`BrianExperiment.persist_data`
                    with BrianExperiment(...) as exp:
                        exp.persist_data[pop_name] = {"pattern":np.ndarray, "sparsity":float}

        :param pop_name: population name for which to compute the similarity threshold
        """
        if not pop_name in self._data["SpikeDeviceGroup"].keys():
            raise ValueError("No such population {pop_name}.")
        if not "pattern" in self._data["persist_data"][pop_name].keys():
            raise ValueError("'pattern' from simulation was not saved.")
        if not "snapshots" in self._analysis["SpikeDeviceGroup"][pop_name].keys():
            raise ValueError("'snapshots' must be analyzed first.")

        # N x pop_size, where N is the number of patterns
        patterns = self._data["persist_data"][pop_name]["pattern"]
        # C x pop_size, where C is the number of cycles
        snapshots = self._analysis["SpikeDeviceGroup"][pop_name]["snapshots"]

        # N x C, (i,j)~ similarity of snapshot from cycle j with pattern i, row i ~ similarities across cycles for pattern i
        distribution = np.zeros((patterns.shape[0], snapshots.shape[0]), dtype=int)
        for i in range(patterns.shape[0]):
            for j in range(snapshots.shape[0]):
                distribution[i, j] = attractor.accuracy(patterns[i], snapshots[j])

        if not "pattern" in self._analysis["SpikeDeviceGroup"][pop_name].keys():
            self._analysis["SpikeDeviceGroup"][pop_name]["pattern"] = {}
        self._analysis["SpikeDeviceGroup"][pop_name]["pattern"][
            "similarity_distribution"
        ] = distribution

