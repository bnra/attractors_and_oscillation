from typing import Any, Dict, Callable, Tuple, Union, List
import numpy as np
from parso import parse
import scipy.signal
import spectrum
import tqdm
import copy

import persistence
from utils import (
    compute_time_interval,
    values_in_interval,
    next_power_of_two,
    restrict_to_interval,
)
import analysis_utils
import parse_equations


class ExperimentAnalysis:
    """
    Analyse data from :class:`BrianExperiment.BrianExperiment`

    Example for analyzing data by all analysis methods

    .. testsetup::

        import numpy as np
        from BrianExperiment import BrianExperiment
        from analysis import ExperimentAnalysis
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
        from analysis import ExperimentAnalysis
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
                self._data["SpikeDeviceGroup"][pn]["device"]["class"]
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
                and self._data["SpikeDeviceGroup"][pn]["device"]["class"]
                == "NeuronPopulation"
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
                self._data["SpikeDeviceGroup"][pn]["device"]["class"]
                == "NeuronPopulation"
            ):
                # note that all data was produced in [t_start, t_end] as this measure depends only on instantaneous_rate
                inst_rate = self._analysis["SpikeDeviceGroup"][pn][
                    "instantaneous_rate"
                ]["value"]

                # # t_start and t_end are now set such that inst_rate contains all samples within [t_start, t_end] (closed bounds)
                # inst_rate, t_start, t_end = restrict_to_interval(
                #     inst_rate, dt, t_start, t_end
                # )

                freq, psd = multitaper_power_spectral_density(inst_rate, dt)

                freq, psd = restrict_frequency(
                    freq, psd, f_lower_bound=f_lower_bound, f_upper_bound=f_upper_bound
                )

                self._analysis["SpikeDeviceGroup"][pn]["psd_complete"] = {
                    "psd": psd,
                    "frequency": {"value": freq, "unit": "Hz"},
                    # "t_start": {"value": t_start / 1000, "unit": "s"},
                    # "t_end": {"value": t_end / 1000, "unit": "s"},
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
                        # "t_start": {"value": t_start / 1000, "unit": "s"},
                        # "t_end": {"value": t_end / 1000, "unit": "s"},
                    }

    def analyze_peaks(
        self, pop_name: List[str] = None, delta: float = 1.0, smoothed: bool = True
    ):
        """
        analyze the peaks (& troughs) of  the population rate of neuron populations to a threshold delta using a time-wise symmetric OR rule
        :param pop_name: population names for which peaks are tb detected
        :param delta: threshold to which peaks are detected
        :param smoothed: whether to analyze the smoothed or instantaneous population rate
        """

        pop_name = (
            list(self._data["SpikeDeviceGroup"].keys())
            if pop_name == None
            else pop_name
        )

        for pn in pop_name:
            if (
                self._data["SpikeDeviceGroup"][pn]["device"]["class"]
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

                mins, maxs = analysis_utils.detect_symmetric_peaks(pop_rate, delta)

                self._analysis["SpikeDeviceGroup"][pn]["peaks"] = {
                    "mins": mins,
                    "maxs": maxs,
                    "delta": delta,
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
                self._data["SpikeDeviceGroup"][pn]["device"]["class"]
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
                not "psd_complete" in self._analysis["SpikeDeviceGroup"][pn].keys()
                and self._data["SpikeDeviceGroup"][pn]["device"]["class"]
                == "NeuronPopulation"
            ):
                psd = self._analysis["SpikeDeviceGroup"][pn]["psd_complete"]["psd"]
                freq = self._analysis["SpikeDeviceGroup"][pn]["psd_complete"][
                    "frequency"
                ]["value"]

                snr = snr(psd, freq, bin_size)

                self._analysis["SpikeDeviceGroup"][pn]["snr"] = {
                    "snr": snr,
                    "bin_size": bin_size,
                }

    def analyze_synchronization_frequency(self):
        """
        synchronization_frequency for populations analyzed in :meth:`ExperimentAnalysis.analyze_power_spectral_density`
        """
        for pn in self._analysis["SpikeDeviceGroup"].keys():
            if (
                not "psd_complete" in self._analysis["SpikeDeviceGroup"][pn].keys()
                and self._data["SpikeDeviceGroup"][pn]["device"]["class"]
                == "NeuronPopulation"
            ):
                psd = self._analysis["SpikeDeviceGroup"][pn]["psd_complete"]["psd"]
                freq = self._analysis["SpikeDeviceGroup"][pn]["psd_complete"][
                    "frequency"
                ]["value"]

                f, p = synchronization_frequency(psd, freq)

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
                not "psd_complete" in self._analysis["SpikeDeviceGroup"][pn].keys()
                and self._data["SpikeDeviceGroup"][pn]["device"]["class"]
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
        conductance_e_e_name: str = "gAMPA_p",
        conductance_i_e_name: str = "gGABA_p",
    ):
        """
        total synaptic conductance for populations pop_e assumes synaptic connectivity to pop_e
        is limited to pop_e -> pop_e and pop_i -> pop_e

        :param pop_e: excitatory population for which total synaptic conductance is computed
        :param pop_i: inhibitory population which connects to pop_e
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

        e_pop_ids = self._data["SpikeDeviceGroup"][pop_e]["ids"]

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
        conductance_e_e = parameters[conductance_e_e_name]["value"]

        synapse_params_e_e = self._data["Synapse"][synapse_e_e]["synapse_params"].load()
        on_pre_e_e = self._data["Synapse"][synapse_e_e]["synapse_params"]["on_pre"]

        neuron_variables_e_e = parse_equations.extract_variables_from_equations(
            on_pre_e_e
        )

        for k, v in neuron_variables_e_e.items():
            synapse_params_e_e[k] = v["neutral_element"]

        parameters_e_e = copy.deepcopy(parameters)
        parameters_e_e.update(synapse_params_e_e)

        # parameters i_e
        conductance_i_e = parameters[conductance_i_e_name]["value"]

        synapse_params_i_e = self._data["Synapse"][synapse_i_e]["synapse_params"].load()
        on_pre_i_e = self._data["Synapse"][synapse_i_e]["synapse_params"]["on_pre"]

        neuron_variables_i_e = parse_equations.extract_variables_from_equations(
            on_pre_i_e
        )

        for k, v in neuron_variables_i_e.items():
            synapse_params_i_e[k] = v["neutral_element"]

        parameters_i_e = parameters
        parameters_i_e.update(synapse_params_i_e)

        # compute total conductance e_e
        eval_on_pre_e_e = parse_equations.evaluate_equations(on_pre_e_e, parameters_e_e)

        syn_input_e_e = eval_on_pre_e_e[synaptic_input_e_e_name]

        (
            targets_e_e,
            sources_per_target_e_e,
            synaptic_input_e_e,
        ) = compute_synaptic_input(source_ids_e_e, target_ids_e_e, syn_input_e_e)

        target_ids_e_e, total_conductance_e_e = effective_total_synaptic_conductance(
            targets_e_e,
            sources_per_target_e_e,
            cell_rate_e,
            synaptic_input_e_e,
            conductance_e_e,
        )

        # compute total conductance i_e
        eval_on_pre_i_e = parse_equations.evaluate_equations(on_pre_i_e, parameters_i_e)

        syn_input_i_e = eval_on_pre_i_e[synaptic_input_i_e_name]

        (
            targets_i_e,
            sources_per_target_i_e,
            synaptic_input_i_e,
        ) = compute_synaptic_input(source_ids_i_e, target_ids_i_e, syn_input_i_e)

        target_ids_i_e, total_conductance_i_e = effective_total_synaptic_conductance(
            targets_i_e,
            sources_per_target_i_e,
            cell_rate_i,
            synaptic_input_i_e,
            conductance_i_e,
        )

        # assumes ids start at 0 and progressively increment
        # inputs from pop_e and pop_i are opposed
        # (a -> b ~ a dependes on b:)
        #  dv_d/dt -> I_ds -> (v_d-v_s),
        #  v_s -> IsynI -> x_GABA
        #  dv_dt -> IsynP -> synP -> x_AMPA
        total_conductance = np.zeros_like(e_pop_ids, dtype=float)
        total_conductance[target_ids_i_e] -= total_conductance_i_e
        total_conductance[target_ids_e_e] += total_conductance_e_e

        self._analysis["SpikeDeviceGroup"][pop_e][
            "total_conductance"
        ] = total_conductance


def gaussian_smoothing(
    instantaneous_rate: np.ndarray,
    window_size: float,
    one_sigma_window: float,
    dt: float,
):
    """
    smoothes the instantaneous rate by window_size around any point ( [-window_size, window_size] ) using a gaussian window
    with one_sigma_window as sigma

    :param instantaneous_rate: instantaneous rate of the population
    :param window_size: size of the window for smoothing in [ms]
    :param one_sigma_window: size of the window encompassing one sigma [ms]
    :param dt: step size of the simulation in [ms] and duration between recordings

    gaussian window implemented acc to https://www.mathworks.com/help/signal/ref/gausswin.html

    (equivalent to brian2.PopulatonRateMonitor.smooth_rate(window='gaussian', width=w) for window_size=2*w and one_sigma_window=w)
    """
    w_n = np.ceil(2 * window_size / dt)
    sigma = np.ceil(one_sigma_window / dt)
    x = np.arange(w_n + 1) - w_n / 2.0
    w = np.exp((-1.0 / 2.0) * (x / sigma) ** 2)

    # ensure the scale of the rates does not change
    return np.convolve(instantaneous_rate, w / np.sum(w), mode="same")


def instantaneous_rate_from_spike_train(
    t: float, dt: float, spike_train: Dict[str, np.ndarray]
):
    """
    computes instantaneous rate from spike train

    :param t: simulation time [ms]
    :param dt: simulation time step [ms]
    :param spike_train: spike trains of individual neurons

    :return: instantaneous firing rate - population average rate for each time step
    """
    spike_events = list(spike_train.values())
    spikes = np.hstack(spike_events) if len(spike_events) > 0 else np.array([])
    vals, counts = np.unique(spikes, return_counts=True)

    rate = np.zeros(values_in_interval(0.0, t, dt))

    idx = np.asarray(np.round(vals / dt), dtype=int)

    # as t,dt in ms rate * 1000/dt ~ Hz (where rate is rate at single step) ~ dt_s = dt / 1000, counts/pop_size / dt_s = counts/pop_size * (1000/dt)
    rate[idx] = counts / len(spike_train.keys()) * 1000 / dt

    return rate


def cell_rate_from_spike_train(
    t_start: float, t_end: float, ids: np.ndarray, spike_train: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    computes cell rate from spike train for each neuron individually

    :param t_start: start time for analysis[ms]
    :param t_end: end time for analysis [ms]
    :param ids: neuron ids of the entire population
    :param spike_train: spike trains of individual neurons

    :return: ids and corresponding cell rate for each cell (time averaged)
    """
    # as t_start, t_end in ms * 1000 ~ Hz

    cell_rates = np.array(
        [
            spike_train[str(i)][
                np.logical_and(
                    spike_train[str(i)] >= t_start, spike_train[str(i)] <= t_end
                )
            ].size
            / (t_end - t_start)
            * 1000
            if str(i) in spike_train.keys()
            else 0.0
            for i in ids
        ]
    )

    return ids, cell_rates


def mt_psd(rate: np.ndarray, dt: float, nfft: int = None):
    """
    Power spectral density of the population rate computed with a multi taper method

    :param rate: population rate
    :param dt: time step / interval of successive measures of the population rate
    :param nfft: length of the output of fft (n-point discrete, where n = nft)
                  - set only if you desire a specific nfft - defaults to 2 ** sp, where sp is smallest num for which rate.size <= 2 ** sp
    :return: frequencies and the power spectral density (at the respective frequencies)
    """

    # equivalent to internal impl (except for capping at 256) as int(np.ceil(np.log2(n))) == int(n-1).bit_length()
    # 2 ** n, where n is smallest power of 2 larger than x.size - fft works best when length is a power of 2
    if nfft == None:
        nfft = 2 ** next_power_of_two(rate.size)

    # sampling frequency [Hz]
    f_s = 1000.0 / dt

    # nyquist frequency [Hz] - max frequency for which signal can be reliable reconstructed ~ f_n is unique minimum (aliasing)
    f_n = f_s / 2.0

    # fft transforms equally spaced samples to same-length equally spaced samples of freq domain at interval 1 / duration of input sequence = 1 / (n / f_s) = f_s / n
    #  - starting at 0.0 for nfft samples we have  nfft * f_s / n ~ f_s and therefore nfft/2 data points in interval [0,f_n]
    frequency = np.linspace(0.0, f_n, nfft // 2)

    # multi taper
    Sk_complex, weights, eigenvalues = spectrum.pmtm(
        rate, NW=2.5, NFFT=nfft, method="eigen"
    )

    # compute the energy spectral density (from complex spectrum)
    Sk = abs(Sk_complex) ** 2

    # average over slepian windows using weights
    spectral = np.mean(Sk * weights, axis=0)[: nfft // 2]

    # compute power spectral density
    spectral = spectral / rate.size

    return frequency, spectral


def multitaper_power_spectral_density(
    rate: np.ndarray,
    dt: float,
    w_sliding: int = None,
    w_step: float = 0.1,
    nfft: int = None,
):
    """
    Power spectral density of the population rate computed with a multi taper method
    computed over the entire time series or for separate (yet overlapping) time intervals using a sliding window
    without padding when parameter w_sliding is set

    :param rate: population rate
    :param dt: time step / interval of successive measures of the population rate
    :param w_sliding: sliding window used for computing psd discretized over time (without padding)
                     - when not set, defaults to computing psd over entire time series
    :param w_step: step size of the sliding window as a fraction of the sliding window size (param w_sliding) - irrelevant when w_sliding not set
    :param nfft: length of the output of fft (n-point discrete, where n = nfft)
                  - set only if you desire a specific nfft, eg to increase the resolution
    :return: frequencies and the power spectral density (for entire time series psd shape: (nfft/2,1); for separate intervals psd shape: (nfft/2, intervals) (at the respective frequencies)
    """

    if w_sliding == None:
        return mt_psd(rate, dt, nfft=nfft)
    else:

        if w_sliding < 1 or w_sliding > rate.size:
            raise ValueError(f"w_sliding must be in [1,rate.size]. Is {w_sliding}")
        if w_step < 0 or w_step > 1:
            raise ValueError(
                f"w_step must be in [0,1] specifying step size as a fraction of sliding window size. Is {w_step}"
            )

        w_step = int(w_sliding * w_step)

        # +1 for intial sliding window size and then rest/w_step intervals on the rest of the sequence
        num_intervals = int((rate.size - w_sliding) / w_step) + 1

        if nfft == None:
            # size of the rate signal in each time interval
            nfft = 2 ** next_power_of_two(w_sliding)

        psd = np.zeros((nfft // 2, num_intervals))
        for i in range(num_intervals):
            frequency, psd[:, i] = mt_psd(
                rate[w_step * i : w_step * i + w_sliding], dt, nfft=nfft
            )
        return frequency, psd


def population_rate_avg_over_time(rate: np.ndarray):
    """
    population rate average over time

    :param rate: population rates over time (shape: (number samples,))
    :return: time average of the population rate
    """
    return np.mean(rate, axis=0)


def synchronization_frequency(
    frequency: np.ndarray, power_spectral_density: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    synchronization frequency : peak frequency of the population rate power spectral density

    :param frequency: frequencies whose power is given by value at respective index of parameter power_spectral_density
    :param power_spectral_density: power of respective frequencies in parameter frequency
    :return: synchronization frequency and its power
    """
    idx = np.argmax(power_spectral_density)
    return frequency[idx], power_spectral_density[idx]


def snr(psd: np.ndarray, frequency: np.ndarray, bin_size: float = 10.0):
    """
    signal-to-noise ratio from power spectral density

    snr = P_signal / P_noise,
    where P_signal is the total power in the bin around the peak frequency
    and P_noise is the total power across the remainder of the spectrum

    :param psd: power spectral density of the signal
    :param frequency: corresponding frequencies of the psd [Hz]
    :param bin_size:  size of the bin [Hz] around the peak frequency used to compute the signal
    """

    peak_idx = np.argmax(psd)
    peak_freq = frequency[peak_idx]

    signal_mask = np.logical_and(
        frequency >= peak_freq - bin_size / 2.0, frequency <= peak_freq + bin_size / 2.0
    )

    return np.sum(psd[signal_mask]) / np.sum(psd[signal_mask == False])


def cross_power_spectral_density():
    """ """
    pass


def restrict_frequency(
    frequency: np.ndarray,
    psd: np.ndarray,
    f_lower_bound: float = None,
    f_upper_bound: float = None,
):
    """
    restrict frequencies and corresponding psd to those for which f_lower_bound <= frequency <= f_upper_bound

    :param frequency: frequencies which are tb restricted
    :param psd: power spectral density that is tb restricted based on the associated frequency value,
            expects the power across frequencies to be on axis 0
    :param f_lower_bound: lower bound on frequency
    :param f_upper_bound: upper bound on frequency
    :return: frequencies and corresponding psd restricted to the interval defined by the bounds
    """

    if f_lower_bound == None and f_upper_bound == None:
        raise ValueError("f_lower_bound and f_upper_bound cannot both be None.")

    idx = np.ones(frequency.size)
    if f_lower_bound != None:
        idx = np.logical_and(idx, frequency >= f_lower_bound)
    if f_upper_bound != None:
        idx = np.logical_and(idx, frequency <= f_upper_bound)
    return frequency[idx], psd[idx]


def compute_synaptic_input(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    synaptic_input: Union[np.ndarray, float],
):
    """
    synaptic input for each distinct id in source_ids by target_id

    :param source_ids: ids representing the source of synaptic connections - each id represents the source neuron of a synapse
    :param target_ids: ids representing the target of synaptic connections - each id represents the target neuron of a synapse
    :param synaptic input: synaptic input by which a variable in the target neuron is modified on spike of source neuron
                            - can be a constant or an array of same size as source_ids and target_ids
    """

    # significant speed up possible using sparse matrix format instead of dicts
    unique_targets = np.unique(target_ids)

    if isinstance(synaptic_input, np.ndarray) and synaptic_input.size > 1:
        # sanity check
        if not (
            synaptic_input.size == source_ids.size
            and source_ids.size == target_ids.size
        ):
            raise ValueError(
                "size of synaptic_input does not match connectivity matrix"
            )

        indices = {tg: np.where(target_ids == tg) for tg in unique_targets}
        sources = {}
        input = {}
        for tg in unique_targets:
            sources[tg] = source_ids[indices[tg]]
            # print(synaptic_input, indices[tg], tg)
            input[tg] = synaptic_input[indices[tg]]

    else:
        sources = {tg: source_ids[np.where(target_ids == tg)] for tg in unique_targets}
        input = np.ones_like(unique_targets, dtype=float) * synaptic_input
    return unique_targets, sources, input


def effective_total_synaptic_conductance(
    target_ids: np.ndarray,
    source_by_target: Dict[int, np.ndarray],
    cell_rate: np.ndarray,
    synaptic_input: Dict[int, np.ndarray],
    conductance: float,
):
    """
    effective total synaptic conductance for a group of synapses on a per target neuron basis

    :param target_ids: target ids for which the total conductance is computed
    :param source_by_target: source ids by the respective target neuron
    :param cell_rate: cell rate of the source population
    :param synaptic_input: synaptic input per source neuron by respective target neuron
            (input by which variables in the target neuron are modified for a spike of the source neuron)
    :param indegree: indegree of the respective target neurons (per neuon)
    :param conductance: target_ids and corresponding total conductance (for this synapse group)
    """
    total_conductance = np.zeros_like(target_ids, dtype=float)
    for i, tg in enumerate(target_ids):
        # assumption the source population starts with id 0 and progressively increases id, ie id = index
        total_conductance[i] = (
            conductance
            * np.sum(synaptic_input[tg] * cell_rate[source_by_target[tg]]).item()
        )
    return target_ids, total_conductance
