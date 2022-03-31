from typing import Dict, Tuple, Union

import scipy.signal as sgnl
import numpy as np
import spectrum
from utils import (
    values_in_interval,
    next_power_of_two,
)


def compute_stimulus_characteristics(stimulus_block_interval: np.ndarray):

    stimulus_onset = stimulus_block_interval[:, 0]
    stimulus_end = stimulus_block_interval[:, 1]

    if not np.all(
        (stimulus_end - stimulus_onset) - (stimulus_end[0] - stimulus_onset[0]) < 1e-5
    ):
        raise ValueError("no constant stimulus length - this is not supported.")
    stimulus_length = (stimulus_end - stimulus_onset)[0]

    # note that the below check guarantees the inter_end_interval is constant across instances of stimulus_ends as well
    # - since stimulus_length is constant
    if stimulus_onset.size < 2 or not np.allclose(
        stimulus_onset[1:] - stimulus_onset[0:-1],
        stimulus_onset[1] - stimulus_onset[0],
    ):
        raise ValueError(
            "a constant stimulus_onset_interval between onsets of the stimulus and"
            + " at least two stimulus presentations are requried. "
            + f"Length of stimulus_presentations {stimulus_onset.size} and stimulus_onset_intervals {stimulus_onset[1:] - stimulus_onset[0:-1]}."
        )
    # note that stimulus_onset[1] - stimulus_onset[0] = stimulus_end[1] - stimulus_end[0]
    inter_presentation_interval = stimulus_onset[1] - stimulus_onset[0]
    return stimulus_onset, stimulus_end, stimulus_length, inter_presentation_interval


def detect_peaks(signal: np.ndarray, dt: float):
    """
    detect peaks of a signal given sampling interval dt with a minimum distance between peaks
    of half the wavelength of the fundamental frequency of the signal
    """
    # determine max freq
    nfft = 2 ** next_power_of_two(signal.size)
    freq, Sk, weights, _ = mt_spectrum(signal, dt, nfft=nfft)

    # print(pop_rate.shape, freq_peak.shape, Sk_peak.shape)

    Sk = Sk[:, : nfft // 2]

    # print(freq_peak.shape, Sk_peak.shape)

    idx = np.logical_and(freq >= 50.0, freq <= 300.0)
    freq, Sk = freq[idx], Sk[:, idx]

    # get argmax of power spectral density
    skc = np.abs(Sk) ** 2
    skc = np.mean(skc * weights, axis=0) / signal.size

    f_sync_idx = np.argmax(skc)
    f_sync = freq[f_sync_idx]

    wave_length = 1 / f_sync * 1000
    sample_distance = int((wave_length / 2) // dt)
    # print(f_sync, wave_length, sample_distance, pop_rate.size)
    peak_idx, _ = sgnl.find_peaks(signal, distance=sample_distance)
    return peak_idx


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
    # time bin spikes
    idx = np.asarray(np.round(spikes / dt), dtype=int)
    idx, counts = np.unique(idx, return_counts=True)

    rate = np.zeros(values_in_interval(0.0, t, dt))

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


def mt_spectrum(rate: np.ndarray, dt: float, nfft: int = None):
    """
    spectrum of the population rate computed with a multi taper method

    :param rate: population rate
    :param dt: time step / interval of successive measures of the population rate
    :param nfft: length of the output of fft (n-point discrete, where n = nft)
                  - set only if you desire a specific nfft - defaults to 2 ** sp, where sp is smallest num for which rate.size <= 2 ** sp
    :return: frequencies, complex spectrum, weights, eigenvalues of multitaper method
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
    return frequency, Sk_complex, weights, eigenvalues


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
    # multi taper
    frequency, Sk_complex, weights, _ = mt_spectrum(rate, dt, nfft)

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


def compute_synaptic_input(source_ids, target_ids, syn_const: Union[float, np.ndarray]):
    """
    synaptic input for each distinct id in source_ids by target_id

    :param source_ids: ids representing the source of synaptic connections - each id represents the source neuron of a synapse
    :param target_ids: ids representing the target of synaptic connections - each id represents the target neuron of a synapse
    :param syn_const: synaptic input constant, input to the target neuron when the source neuron spikes - either a scalar value (same for all synapses) or one value per synaptic connection
    """
    unique_targets = np.sort(np.unique(target_ids))
    # indices in ascending order
    unique_sources = np.sort(np.unique(source_ids))

    # Targets x Sources - row represents all synaptic inputs from all sources to a target
    synaptic_input = np.zeros((unique_targets.size, unique_sources.size), dtype=float)

    # broadcasts if syn_const is scalar
    synaptic_input[target_ids, source_ids] = syn_const

    return unique_targets, unique_sources, synaptic_input


def synaptic_conductance(
    target_ids: np.ndarray,
    source_ids: np.ndarray,
    cell_rate: np.ndarray,
    synaptic_input: np.ndarray,
    conductance: float,
):
    """
    total synaptic conductance for a specific synaptic input type for a group of synapses on a per target neuron basis

    :param target_ids: target ids for which the total conductance is computed (unique targets) sorted in ascending order
    :param source_ids: source ids (unique sources) sorted in ascending order
    :param cell_rate: cell rate of the source population
    :param synaptic_input: synaptic input per target and source neuron (targets x source)
    :param conductance: conductance of the respective synaptic input type
    :return: target_ids and corresponding total conductance of the respective synaptic input (for this synapse group)
    """
    # synaptic input: Targets x Sources @  cell_rate[source_ids]: Sources x 1 -> Targets x 1
    return (
        target_ids,
        conductance * synaptic_input @ cell_rate[source_ids],
    )


def effective_total_synaptic_conductance(
    source_ids_e_e: np.ndarray,
    target_ids_e_e: np.ndarray,
    source_ids_i_e: np.ndarray,
    target_ids_i_e: np.ndarray,
    cell_rate_e: np.ndarray,
    cell_rate_i: np.ndarray,
    syn_const_e_e: float,
    syn_const_i_e: float,
    conductance_e_e: float,
    conductance_i_e: float,
):

    """
    effective total synaptic conductance for a group of synapses on a per target neuron basis,
    e refers to the excitatory population and i to the inhibitory population in an EI-Network

    :param source_ids_e_e: ids representing the source of e-e synaptic connections - each id represents the source neuron of a synapse
    :param target_ids_e_e: ids representing the target of e-e synaptic connections - each id represents the target neuron of a synapse
    :param source_ids_i_e: ids representing the source of i-e synaptic connections - each id represents the source neuron of a synapse
    :param target_ids_i_e: ids representing the target of i-e synaptic connections - each id represents the target neuron of a synapse

    :param cell_rate_e: cell rate of the excitatory population
    :param cell_rate_i: cell rate of the inhibitory population

    :param syn_const_e_e: synaptic input (constant) to the target neurons when the source neuron spikes for e-e synapses
                          - either a scalar value (same for all synapses) or one value per synaptic connection
    :param syn_const_i_e: synaptic input (constant) to the target neurons when the source neuron spikes for i-e synapses
                          - either a scalar value (same for all synapses) or one value per synaptic connection
    :param conductance_e_e: conductance for e-e synapses
    :param conductance_i_e: conductance for i-e synapses

    :return: effective total synaptic conductance per neuron of the entire e population
    """

    targets_e_e, sources_e_e, synaptic_input_e_e = compute_synaptic_input(
        source_ids_e_e, target_ids_e_e, syn_const_e_e
    )
    targets_i_e, sources_i_e, synaptic_input_i_e = compute_synaptic_input(
        source_ids_i_e, target_ids_i_e, syn_const_i_e
    )

    target_ids_e_e, total_conductance_e_e = synaptic_conductance(
        targets_e_e,
        sources_e_e,
        cell_rate_e,
        synaptic_input_e_e,
        conductance_e_e,
    )
    target_ids_i_e, total_conductance_i_e = synaptic_conductance(
        targets_i_e,
        sources_i_e,
        cell_rate_i,
        synaptic_input_i_e,
        conductance_i_e,
    )

    # assumes ids start at 0 and progressively increment
    # inputs from pop_e (excitatory) and pop_i (inhibitory) are opposed
    # note cell_rate.size ~ population size

    total_conductance = np.zeros(cell_rate_e.size, dtype=float)
    total_conductance[target_ids_i_e] -= total_conductance_i_e
    total_conductance[target_ids_e_e] += total_conductance_e_e

    return total_conductance
