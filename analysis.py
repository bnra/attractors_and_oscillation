from typing import Dict, Callable, Tuple, Union, List
import numpy as np
import scipy.signal
import spectrum
import tqdm

import persistence
from utils import (
    compute_time_interval,
    values_in_interval,
    next_power_of_two,
    restrict_to_interval,
)
import analysis_utils


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
                }


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
    t: float, dt: float, spike_train: Dict[str, float]
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
    # print(
    #     f"c:{counts[0]}, pop size: {len(spike_train.keys())}, dt:{dt}, rate:{counts[0] / len(spike_train.keys()) * 1000 / dt}"
    # )
    rate[idx] = counts / len(spike_train.keys()) * 1000 / dt

    return rate


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
