import scipy.sparse as sp
from scipy.special import factorial
from typing import List, Tuple, Dict, Union
import numpy as np
import analysis
from mpmath import bernoulli


def resolve_time_interval(
    stimulus_onset: np.ndarray,
    inter_presentation_interval: float,
    stimulus_length: float,
):
    """
    compute time intervals relative to the onset of stimulus presentations [stimulus_onset - inter_presentation_interval + stimulus_length, stimulus_onset + inter_presentation_interval],
    given stimulus_onset, inter_presentation_interval and stimulus_length

    :param stimulus_onset: set of onsets of stimulus presentations [ms]
    :param inter_presentation_interval: interval between two stimulus presentation onsets or ceasures equivalently [ms]
    :param stimulus_length: length of stimulus [ms]
    :return: time intervals relative to the onset of stimulus presentations [stimulus_onset - inter_presentation_interval + stimulus_length, stimulus_onset + inter_presentation_interval]
    """
    return (
        stimulus_onset - inter_presentation_interval + stimulus_length,
        stimulus_onset + inter_presentation_interval,
    )


def resolve_spike_times(
    stimulus_onset: np.ndarray,
    inter_presentation_interval: float,
    stimulus_length: float,
):
    """
    resolve spike times relative to the onset of stimulus presentations [stimulus_onset - inter_presentation_interval + stimulus_length, stimulus_onset + inter_presentation_interval],
    given stimulus_onset, inter_presentation_interval and stimulus_length

    :param stimulus_onset: set of onsets of stimulus presentations [ms]
    :param inter_presentation_interval: interval between two stimulus presentation onsets or ceasures equivalently [ms]
    :param stimulus_length: length of stimulus [ms]
    :return: spike times relative to stimulus onset
    """

    zauber = 3

    return (
        stimulus_onset - inter_presentation_interval + stimulus_length,
        stimulus_onset + inter_presentation_interval,
    )


def resolve_snapshots(
    troughs: np.ndarray,
    inter_presentation_interval: float,
    first_stimulus_onset: float,
):
    """
    resolve snapshots relative to the onset of stimulus presentations,
    a snapshot is boolean spike mask over all cells indicating which cells spike over the interval between two troughs
    - the mid point between two troughs will be used as standin for the spike train of all spiking cells during the snapshot

    :param troughs: troughs of population rate [ms] - allow computing the start and end time of each snapshot (see :func:`attractor.extract_snapshots`)
    :param inter_presentation_interval: interval between two stimulus presentation onsets or ceasures equivalently [ms]
    :return: resolved spike times relative to stimulus onset
    """
    snap_beg, snap_end = troughs[:-1], troughs[1:]
    snap_spike = (snap_end - snap_beg) / 2 + snap_beg
    return (snap_spike - first_stimulus_onset) % inter_presentation_interval


def separate_presentation_cycles(
    troughs: np.ndarray,
    peaks: np.ndarray,
    stimulus_onset: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    separate presentation cycles according to the distance of the stimulus onset to the trough or peak, respectively
    ideally trough group: [t_trough-w/4, t_trough + w/4) or ii) peak group:[t_peak-w/4, t_peak + w/4).
    effectively: assign presentation cycle to group trough, peak for distance from stimulus onset to trough < peak, peak < trough, respectively
    where w refers to the wavelength, t_trough is the time point of any trough and therefore marks the beginning and ending of a snapshot, respectively.


    :param troughs: troughs of population rate [ms] (C+1,) where C is # snapshots
    :param peaks: peaks of population rate [ms]
    :param stimulus_onset: time points of stimulus onset [ms](S,) where S is the number of stimulus presentations
    :param stimulus_length: length of stimulus_presentation [ms]
    :param inter_presentation_interval: interval between two stimulus presentation onsets or ceasures equivalently [ms]
    :return: indices of trough cycles and peak cycles (S,) where S is the number of stimulus presentations
    """

    # proximity
    proximity_troughs = np.zeros_like(stimulus_onset, dtype=int)
    proximity_peaks = np.zeros_like(stimulus_onset, dtype=int)

    for i, so in enumerate(stimulus_onset):
        proximity_troughs[i] = np.argmin(np.abs(troughs - so))
        proximity_peaks[i] = np.argmin(np.abs(peaks - so))

    # 2 cases: i) distance to trough is smaller ii) if distance equal (abs(peak-so) == abs(trough-so)), then two cases: a) peak < so < trough and b) trough < so < peak
    # -> trough cycle cases: i) and iia)
    trough_cycle = np.logical_or(
        np.abs(troughs[proximity_troughs] - stimulus_onset)
        < np.abs(peaks[proximity_peaks] - stimulus_onset),
        np.logical_and(
            np.abs(troughs[proximity_troughs] - stimulus_onset)
            == np.abs(peaks[proximity_peaks] - stimulus_onset),
            troughs[proximity_troughs] > stimulus_onset,
        ),
    )
    peak_cycle = np.logical_not(trough_cycle)

    return np.nonzero(trough_cycle)[0], np.nonzero(peak_cycle)[0]


def separate_snapshots(
    troughs: np.ndarray,
    stimulus_end: np.ndarray,
    stimulus_length: float,
    inter_presentation_interval: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    separate snapshots according to the distance of the time point of ceasure of the stimulus presentation to the time point at which the next snapshot starts
    - a snapshot is boolean spike mask over all cells indicating which cells spike over the interval between two troughs -into three groups:
    i) trough group - w/4 around any trough [t_trough-w/4, t_trough + w/4) ii) peak group - w/4 around any peak [t_trough - 3/4*w, t_trough-1/4*w)
    iii) null group - all snapshots not in i) and not in ii) ie. snapshots during which stimulus presentation occurs and does not end before t_snap_0 + w/4
    (with t_snap_0 the time point at which snapshot starts)
    where w refers to the wavelength, t_trough is the time point of any trough and therefore marks the beginning and ending of a snapshot, respectively.


    :param troughs: troughs of population rate [ms] (C+1,) where C is # snapshots - allow computing the start and end time of each snapshot (see :func:`attractor.extract_snapshots`)
    :param stimulus_end: time points of ceasure of the stimulus presentations [ms](S,) where S is the number of stimulus presentations
    :param stimulus_length: length of stimulus_presentation [ms]
    :param inter_presentation_interval: interval between two stimulus presentation onsets or ceasures equivalently [ms]
    :return: indices of peak group, indices of trough group, indices of null group for the vector of snapshots (C,)
             and indices of peak cycle, trough cycle and null cycle (S,) where S is the number of stimulus presentations
    """

    # compute the wave_length as duration over snapshots
    wave_length = (troughs[-1] - troughs[0]) / (troughs.size - 1)
    # snapshots are defined as trough to trough: ie snapshot i occurs in interval [troughs[i], troughs[i+1]] and encloses a peak within
    snap_beg, snap_end = troughs[:-1], troughs[1:]

    trough_group = np.zeros(troughs.size - 1, dtype=bool)
    peak_group = np.zeros(troughs.size - 1, dtype=bool)

    trough_cycle = np.zeros_like(stimulus_end, dtype=bool)
    peak_cycle = np.zeros_like(stimulus_end, dtype=bool)

    for i, se in enumerate(stimulus_end):
        # most recent snap prior to stimulus end
        t_prev = (
            troughs[np.nonzero(troughs < se)[0]][-1].item()
            if np.sum(troughs < se) > 0
            else None
        )
        # first snapshot after stimulus end
        t_next = (
            troughs[np.nonzero(troughs >= se)[0]][0].item()
            if np.sum(troughs >= se) > 0
            else None
        )

        # edge case where stimulus before first trough or after last ie not in snapshot
        if t_prev == None or t_next == None:
            continue

        # print(f"t_prev {t_prev:.2f}, t_next {t_next:.2f}, se {se:.2f}")
        # trough group
        # ideally equiv to t_prev + se < 1/4 * w
        if t_next - se > 3 / 4 * wave_length:
            # print("trough t_prev")
            trough_group[
                np.logical_and(
                    snap_beg >= t_prev,
                    snap_end < se - stimulus_length + inter_presentation_interval,
                )
            ] = True
            trough_cycle[i] = True
        else:

            if t_next - se <= wave_length / 4:
                # print("trough t_next")
                trough_group[
                    np.logical_and(
                        snap_beg >= t_next,
                        snap_end < se - stimulus_length + inter_presentation_interval,
                    )
                ] = True
                trough_cycle[i] = True

            else:
                # print(f"peak t_next: {np.nonzero(np.logical_and(snap_beg >= t_next, snap_end < se - stimulus_length + inter_onset_interval))[0]}")
                peak_group[
                    np.logical_and(
                        snap_beg >= t_next,
                        snap_end < se - stimulus_length + inter_presentation_interval,
                    )
                ] = True
                peak_cycle[i] = True

    return (
        np.nonzero(trough_group)[0],
        np.nonzero(peak_group)[0],
        np.nonzero(np.logical_not(np.logical_or(trough_group, peak_group)))[0],
        np.nonzero(trough_cycle)[0],
        np.nonzero(peak_cycle)[0],
        np.nonzero(np.logical_not(np.logical_or(trough_cycle, peak_cycle)))[0],
    )


def fraction_significant_snapshots(
    pvalue: np.ndarray,
    stimulus_pattern: np.ndarray,
    pattern: np.ndarray,
    significance: float = 0.05,
) -> Tuple[float, np.ndarray]:
    """
    compute fraction of significant snapshots

    :param pvalue: pvalue for each snapshot (N, C) where C is the number of snapshots and N the number of patterns
    :param stimulus_pattern: stimulus pattern used to stimulate the network (pattern_length,) and one of the patterns in parameter 'pattern'
    :param pattern: patterns from which the scaling matrix (weights) is computed (N x pattern_length,)
    :param significance: significance level at which the fration of significant snapshots is computed
    :return: fraction of significant snapshots and corresponding pvalues (C,), where C is the number of snapshots
    """
    if pvalue.size > 0:
        # shape: (pattern-length,)
        pattern_idx = np.all(
            pattern
            == np.tile(stimulus_pattern, pattern.shape[0]).reshape(
                pattern.shape[0], -1
            ),
            axis=1,
        )
        # pattern_idx is a one hot vector of length N, pvals dims: (N x C)
        pvalue = pvalue[pattern_idx]
        return (np.sum(pvalue <= significance) / pvalue.size, pvalue)
    else:
        return (0.0, np.array([]))


def indices_snapshots_blocked_stimulus(
    pvalue: np.ndarray,
    stimulus_pattern: np.ndarray,
    pattern: np.ndarray,
    troughs: np.ndarray,
    peaks: np.ndarray,
    stimulus_onset: np.ndarray,
    stimulus_length: float,
) -> Tuple[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    compute fraction of significant snapshots for blocked stimulus given a stimulus length and stimulus onset times for windows
    [-stimulus_length,t0],[t0,t0+stimulus_length],[t0+stimulus_length, t0+2*stimulus_length],
    note that the stimulus occurs in interval t_beg = t0 and t_end = t0 + stimulus_length
    - a snapshot ([t_snap_beg, t_snap_end]) is considered if for its enclosed peak (t_peak)
        there is an interval ([t_beg, t_end]) such that t_peak < t_end and t_peak >= t_beg
        or t_snap_beg<= t_beg and t_snap_end >= t_end (ie snapshot encloses interval)


    :param pvalue: pvalue for each snapshot (N, C) where C is the number of snapshots and N the number of patterns
    :param stimulus_pattern: stimulus pattern used to stimulate the network (pattern_length,) and one of the patterns in parameter 'pattern'
    :param pattern: patterns from which the scaling matrix (weights) is computed (N x pattern_length,)
    :param troughs: troughs of population rate [ms] - allow computing the start and end time of each snapshot (see :func:`attractor.extract_snapshots`)
    :param peaks: peaks of population rate [ms] used for determining the inclusion of snapshots that start before or end after an interval (edge case)
    :param stimulus_onset: onset times of the stimulus [ms]
    :param stimulus_length: length of the stimulus [ms]
    :param significance: significance level at which the fration of significant snapshots is computed
    :return: fraction of significant snapshots within time windows and corresponding pvalues (C,), where C is the number of snapshots
            ([t0-stimulus_length,t0], [t0,t0+stimulus_length], [t0+stimulus_length, t0+2*stimulus_length])
    """

    return [
        indices_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            list(zip(t_beg, t_end)),
        )
        for t_beg, t_end in [
            (stimulus_onset - stimulus_length, stimulus_onset),
            (stimulus_onset, stimulus_onset + stimulus_length),
            (
                stimulus_onset + stimulus_length,
                stimulus_onset + 2 * stimulus_length,
            ),
        ]
    ]


def fraction_significant_snapshots_blocked_stimulus(
    pvalue: np.ndarray,
    stimulus_pattern: np.ndarray,
    pattern: np.ndarray,
    troughs: np.ndarray,
    peaks: np.ndarray,
    stimulus_onset: np.ndarray,
    stimulus_length: float,
    significance: float = 0.05,
) -> Tuple[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    compute fraction of significant snapshots for blocked stimulus given a stimulus length and stimulus onset times for windows
    [-stimulus_length,t0],[t0,t0+stimulus_length],[t0+stimulus_length, t0+2*stimulus_length],
    note that the stimulus occurs in interval t_beg = t0 and t_end = t0 + stimulus_length
    - a snapshot ([t_snap_beg, t_snap_end]) is considered if for its enclosed peak (t_peak)
        there is an interval ([t_beg, t_end]) such that t_peak < t_end and t_peak >= t_beg
        or t_snap_beg<= t_beg and t_snap_end >= t_end (ie snapshot encloses interval)


    :param pvalue: pvalue for each snapshot (N, C) where C is the number of snapshots and N the number of patterns
    :param stimulus_pattern: stimulus pattern used to stimulate the network (pattern_length,) and one of the patterns in parameter 'pattern'
    :param pattern: patterns from which the scaling matrix (weights) is computed (N x pattern_length,)
    :param troughs: troughs of population rate [ms] - allow computing the start and end time of each snapshot (see :func:`attractor.extract_snapshots`)
    :param peaks: peaks of population rate [ms] used for determining the inclusion of snapshots that start before or end after an interval (edge case)
    :param stimulus_onset: onset times of the stimulus [ms]
    :param stimulus_length: length of the stimulus [ms]
    :param significance: significance level at which the fration of significant snapshots is computed
    :return: fraction of significant snapshots within time windows and corresponding pvalues (C,), where C is the number of snapshots
            ([t0-stimulus_length,t0], [t0,t0+stimulus_length], [t0+stimulus_length, t0+2*stimulus_length])
    """

    indices = indices_snapshots_blocked_stimulus(
        pvalue,
        stimulus_pattern,
        pattern,
        troughs,
        peaks,
        stimulus_onset,
        stimulus_length,
    )
    pvalue = [np.hstack([pvalue[ix] for ix in idx]) for idx in indices]
    return tuple(
        zip(
            *[
                (np.sum(pv <= significance) / pv.size, pv)
                if pv.size > 0
                else (0.0, np.ndarray([]))
                for pv in pvalue
            ]
        )
    )


def fraction_significant_snapshots_blocked_stimulus_detailed(
    pvalue: np.ndarray,
    stimulus_pattern: np.ndarray,
    pattern: np.ndarray,
    troughs: np.ndarray,
    peaks: np.ndarray,
    stimulus_onset: np.ndarray,
    stimulus_length: float,
    significance: float = 0.05,
) -> Tuple[
    Tuple[float, float, float, float, float],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    compute fraction of significant snapshots for blocked stimulus given a stimulus length and stimulus onset times for windows
    ([t0-stimulus_length,t0-stimulus_length/2], [t0-stimulus_length/2, t0], [t0,t0+stimulus_length], [t0+stimulus_length, t0+1.5*stimulus_length],[t0+1.5*stimulus_length,t0+2*stimulus_length])],
    note that the stimulus occurs in interval t_beg = t0 and t_end = t0 + stimulus_length
    - a snapshot ([t_snap_beg, t_snap_end]) is considered if for its enclosed peak (t_peak)
        there is an interval ([t_beg, t_end]) such that t_peak < t_end and t_peak >= t_beg
        or t_snap_beg<= t_beg and t_snap_end >= t_end (ie snapshot encloses interval)


    :param pvalue: pvalue for each snapshot (N, C) where C is the number of snapshots and N the number of patterns
    :param stimulus_pattern: stimulus pattern used to stimulate the network (pattern_length,) and one of the patterns in parameter 'pattern'
    :param pattern: patterns from which the scaling matrix (weights) is computed (N x pattern_length,)
    :param troughs: troughs of population rate [ms] - allow computing the start and end time of each snapshot (see :func:`attractor.extract_snapshots`)
    :param peaks: peaks of population rate [ms] used for determining the inclusion of snapshots that start before or end after an interval (edge case)
    :param stimulus_onset: onset times of the stimulus [ms]
    :param stimulus_length: length of the stimulus [ms]
    :param significance: significance level at which the fration of significant snapshots is computed
    :return: fraction of significant snapshots within time windows and corresponding pvalues (C,), where C is the number of snapshots
    ([t0-stimulus_length,t0-stimulus_length/2], [t0-stimulus_length/2, t0], [t0,t0+stimulus_length], [t0+stimulus_length, t0+1.5*stimulus_length],[t0+1.5*stimulus_length,t0+2*stimulus_length])
    """

    return [
        fraction_significant_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            list(zip(t_beg, t_end)),
            significance,
        )
        for t_beg, t_end in [
            (
                stimulus_onset - stimulus_length,
                stimulus_onset - stimulus_length / 2,
            ),
            (stimulus_onset - stimulus_length / 2, stimulus_onset),
            (stimulus_onset, stimulus_onset + stimulus_length),
            (
                stimulus_onset + stimulus_length,
                stimulus_onset + 1.5 * stimulus_length,
            ),
            (
                stimulus_onset + 1.5 * stimulus_length,
                stimulus_onset + 2.0 * stimulus_length,
            ),
        ]
    ]


def indices_snapshots_blocked_stimulus_sliding_window(
    pvalue: np.ndarray,
    stimulus_pattern: np.ndarray,
    pattern: np.ndarray,
    troughs: np.ndarray,
    peaks: np.ndarray,
    stimulus_onset: float,
    t_end: float,
    inter_onset_interval: float,
    window_length: float,
    window_step: float,
) -> Tuple[Tuple[float, ...], Tuple[np.ndarray, ...]]:
    """
    compute fraction of significant snapshots for blocked stimulus given a stimulus length and stimulus onset times for a sliding window
    of length window_length and with step window_step
    note that the stimulus occurs in interval t_beg = t0 and t_end = t0 + stimulus_length
    - a snapshot ([t_snap_beg, t_snap_end]) is considered if for its enclosed peak (t_peak)
        there is an interval ([t_beg, t_end]) such that t_peak < t_end and t_peak >= t_beg
        or t_snap_beg<= t_beg and t_snap_end >= t_end (ie snapshot encloses interval)


    :param pvalue: pvalue for each snapshot (N, C) where C is the number of snapshots and N the number of patterns
    :param stimulus_pattern: stimulus pattern used to stimulate the network (pattern_length,) and one of the patterns in parameter 'pattern'
    :param pattern: patterns from which the scaling matrix (weights) is computed (N x pattern_length,)
    :param troughs: troughs of population rate [ms] - allow computing the start and end time of each snapshot (see :func:`attractor.extract_snapshots`)
    :param peaks: peaks of population rate [ms] used for determining the inclusion of snapshots that start before or end after an interval (edge case)
    :param stimulus_onset: onset time of the first stimulus presentation [ms]
    :param t_end: end of the simulation
    :param inter_onset_interval: interval time between the onsets of any two subsequent stimulus presentations [ms]
    :param window_length: length of the sliding window
    :param window_step: step size of the sliding window
    :return: indices of significant snapshots within time windows
            [stimulus_onset, stimulus_onset + window_step, stimulus_onset + 2 * window_step, ..., stimulus_onset + inter_onset_interval]
    """

    if (
        # to deal with limited floating point precision - either close to 0 or window_step due to modulo and edge case around 0
        not (
            inter_onset_interval % window_step < 1e-10
            or inter_onset_interval % window_step - window_step < 1e-10
        )
        and inter_onset_interval / window_step >= 1.0
    ):
        raise ValueError(
            f"'inter_onset_interval' must be evenly divisible by 'window_step'. Result of abs({inter_onset_interval}%{window_step}) is {inter_onset_interval % window_step}."
        )

    num_windows = round(inter_onset_interval / window_step)


    # note that we will start with stimulus_onset (of first stimulus presentation) and end with final trough (end of final snap)
    # + 1e-5 ensures troughs[-1] is included if multiple of ioi
    onsets = np.arange(stimulus_onset, troughs[-1] + 1e-5, inter_onset_interval)
    # onsets must have a relaxation period of (inter_onset_interval - stimulus_length) else they do not occur - edge case (final one)
    onsets = onsets[onsets <= t_end - inter_onset_interval]

    return [
        indices_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            list(zip(t_beg, t_end)),
        )
        for t_beg, t_end in [
            # beginning and end of each window: starting at onset: [...,(onset+i*ws-window_length/2, onset+i*ws+window_length/2),...]
            # where i is the ith window centered around time point onset + i*ws and ws is the window_step
            (
                onsets + i * window_step - window_length / 2,
                onsets + i * window_step + window_length / 2,
            )
            for i in range(num_windows)
        ]
    ]


def fraction_significant_snapshots_blocked_stimulus_sliding_window(
    pvalue: np.ndarray,
    stimulus_pattern: np.ndarray,
    pattern: np.ndarray,
    troughs: np.ndarray,
    peaks: np.ndarray,
    stimulus_onset: float,
    t_end: float,
    inter_onset_interval: float,
    window_length: float,
    window_step: float,
    significance: float = 0.05,
) -> Tuple[Tuple[float, ...], Tuple[np.ndarray, ...]]:
    """
    compute fraction of significant snapshots for blocked stimulus given a stimulus length and stimulus onset times for a sliding window
    of length window_length and with step window_step
    note that the stimulus occurs in interval t_beg = t0 and t_end = t0 + stimulus_length
    - a snapshot ([t_snap_beg, t_snap_end]) is considered if for its enclosed peak (t_peak)
        there is an interval ([t_beg, t_end]) such that t_peak < t_end and t_peak >= t_beg
        or t_snap_beg<= t_beg and t_snap_end >= t_end (ie snapshot encloses interval)


    :param pvalue: pvalue for each snapshot (N, C) where C is the number of snapshots and N the number of patterns
    :param stimulus_pattern: stimulus pattern used to stimulate the network (pattern_length,) and one of the patterns in parameter 'pattern'
    :param pattern: patterns from which the scaling matrix (weights) is computed (N x pattern_length,)
    :param troughs: troughs of population rate [ms] - allow computing the start and end time of each snapshot (see :func:`attractor.extract_snapshots`)
    :param peaks: peaks of population rate [ms] used for determining the inclusion of snapshots that start before or end after an interval (edge case)
    :param stimulus_onset: onset time of the first stimulus presentation [ms]
    :param t_end: end of the simulation
    :param inter_onset_interval: interval time between the onsets of any two subsequent stimulus presentations [ms]
    :param window_length: length of the sliding window
    :param window_step: step size of the sliding window
    :param significance: significance level at which the fration of significant snapshots is computed
    :return: fraction of significant snapshots within time windows and corresponding pvalues (C,), where C is the number of snapshots
            [stimulus_onset, stimulus_onset + window_step, stimulus_onset + 2 * window_step, ..., stimulus_onset + inter_onset_interval]
    """
    indices = indices_snapshots_blocked_stimulus_sliding_window(
        pvalue,
        stimulus_pattern,
        pattern,
        troughs,
        peaks,
        stimulus_onset,
        t_end,
        inter_onset_interval,
        window_length,
        window_step,
    )

    pvalue = [np.hstack([pvalue[ix] for ix in idx]) for idx in indices]
    return tuple(
        zip(
            *[
                (np.sum(pv <= significance) / pv.size, pv)
                if pv.size > 0
                else (0.0, np.ndarray([]))
                for pv in pvalue
            ]
        )
    )


def fraction_significant_snapshots_across_intervals(
    pvalue: np.ndarray,
    stimulus_pattern: np.ndarray,
    pattern: np.ndarray,
    troughs: np.ndarray,
    peaks: np.ndarray,
    interval: List[Tuple[float, float]],
    significance: float = 0.05,
) -> Tuple[float, np.ndarray]:
    """
    compute fraction of significant snapshots across a set of intervals
    - a snapshot ([t_snap_beg, t_snap_end]) is considered if for its enclosed peak (t_peak)
        there is an interval ([t_beg, t_end]) such that t_peak < t_end and t_peak >= t_beg
        or t_snap_beg<= t_beg and t_snap_end >= t_end (ie snapshot encloses interval)


    :param pvalue: pvalue for each snapshot (N, C) where C is the number of snapshots and N the number of patterns
    :param stimulus_pattern: stimulus pattern used to stimulate the network (pattern_length,) and one of the patterns in parameter 'pattern'
    :param pattern: patterns from which the scaling matrix (weights) is computed (N x pattern_length,)
    :param troughs: troughs of population rate [ms] (C+1,)- allow computing the start and end time of each snapshot (see :func:`attractor.extract_snapshots`)
    :param peaks: peaks of population rate [ms] used for determining the inclusion of snapshots that start before or end after an interval (edge case)
    :param interval: set of intervals defining which snapshots are considered for the computation: a snapshot ([t_snap_beg, t_snap_end] - def. by
                parameter 'troughs') enclosing peak (t_peak - given in peaks) is considered
                if there is an interval ([t_beg, t_end]) such that t_peak < t_end and t_peak >= t_beg
                or t_snap_beg<= t_beg and t_snap_end >= t_end (ie snapshot encloses interval)
    :param significance: significance level at which the fration of significant snapshots is computed
    :return: fraction of significant snapshots within intervals and corresponding pvalues (C,), where C is the number of snapshots
    """

    idx = indices_snapshots_across_intervals(
        pvalue, stimulus_pattern, pattern, troughs, peaks, interval
    )
    pvalue = np.hstack([pvalue[ix] for ix in idx])
    return (
        (np.sum(pvalue <= significance) / pvalue.size, pvalue)
        if pvalue.size > 0
        else (0.0, np.ndarray([]))
    )


def indices_snapshots_across_intervals(
    pvalue: np.ndarray,
    stimulus_pattern: np.ndarray,
    pattern: np.ndarray,
    troughs: np.ndarray,
    peaks: np.ndarray,
    interval: List[Tuple[float, float]],
) -> Tuple[float, np.ndarray]:
    """
    extract idx of significant snapshots across a set of intervals
    - a snapshot ([t_snap_beg, t_snap_end]) is considered if for its enclosed peak (t_peak)
        there is an interval ([t_beg, t_end]) such that t_peak < t_end and t_peak >= t_beg
        or t_snap_beg<= t_beg and t_snap_end >= t_end (ie snapshot encloses interval)


    :param pvalue: pvalue for each snapshot (N, C) where C is the number of snapshots and N the number of patterns
    :param stimulus_pattern: stimulus pattern used to stimulate the network (pattern_length,) and one of the patterns in parameter 'pattern'
    :param pattern: patterns from which the scaling matrix (weights) is computed (N x pattern_length,)
    :param troughs: troughs of population rate [ms] (C+1,)- allow computing the start and end time of each snapshot (see :func:`attractor.extract_snapshots`)
    :param peaks: peaks of population rate [ms] used for determining the inclusion of snapshots that start before or end after an interval (edge case)
    :param interval: set of intervals defining which snapshots are considered for the computation: a snapshot ([t_snap_beg, t_snap_end] - def. by
                parameter 'troughs') enclosing peak (t_peak - given in peaks) is considered
                if there is an interval ([t_beg, t_end]) such that t_peak < t_end and t_peak >= t_beg
                or t_snap_beg <= t_beg and t_snap_end >= t_end (ie snapshot encloses interval)
    :return: set of boolean mask of pvalues (N,C) indexing all pvalues of stimulus_pattern of snapshots within an interval for each
    """
    # ensure that intervals do not overlap
    # dim (num_its, 2): its[:,0] beginning of its, its[:,1] ends of its
    its = np.array(interval)
    for i, (t_beg, t_end) in enumerate(interval):
        it = np.vstack((its[0:i], its[i + 1 :]))
        # in case of an overlap either i) 'it' is contained in other interval or ii) 'it' starts before other interval and ends within the other interval
        #    - note: the mirror cases to i) 'it' encloses an interval and ii) 'it' starts within some interval and lasts beyond ending of the interval
        #      are covered by the first two cases when considering the other interval as the reference point (~ renaming)
        overlaps = np.logical_or(
            # contained case
            np.logical_and(t_beg >= it[:, 0], t_end <= it[:, 1]),
            # actual overlap left ie. interval starts before sum other interval and they overlap (ie. t_end > beginning of some interval, t_beg < ending of some interval)
            np.logical_and(t_beg < it[:, 0], t_end > it[:, 0]),
        )
        if np.any(overlaps):
            raise ValueError(
                f"Overlaps detected for interval [{t_beg}, {t_end}]: with intervals {np.argwhere(overlaps)[0]}: {it[overlaps]}"
            )

    # snapshots are defined as trough to trough: ie snapshot i occurs in interval [troughs[i], troughs[i+1]] and encloses a peak within
    snap_beg, snap_end = [
        np.array(e)
        for e in zip(*[(troughs[i], troughs[i + 1]) for i in range(troughs.size - 1)])
    ]



    # for each interval extract enclosed peak
    snap_peak_idx = [
        np.where(np.logical_and(peaks >= beg, peaks <= end))[0]
        for beg, end in zip(snap_beg, snap_end)
    ]
    # use first index of peak - note that there should be exactly one enclosed peak as snapshots are defined as trough to trough
    # in case there is no peak detected btw two troughs - use the midpoint btw the two troughs
    snap_peak = np.array(
        [
            peaks[idx[0]]
            if len(idx) != 0
            else (troughs[i + 1] - troughs[i]) / 2 + troughs[i]
            for i, idx in enumerate(snap_peak_idx)
        ]
    )

    index = np.zeros_like(pvalue, dtype=bool)
    if pvalue.size > 0:
        # extract pvalues of stimulus pattern
        # shape: (pattern-length,)
        pattern_idx = np.all(
            pattern
            == np.tile(stimulus_pattern, pattern.shape[0]).reshape(
                pattern.shape[0], -1
            ),
            axis=1,
        )
        # pattern_idx is a one hot vector of length N, pvals dims: (N x C)
        # pvalue (N x C) -> (C,) assuming there is exactly one matching pattern
        pvalue = pvalue[pattern_idx].squeeze()

        if len(pvalue.shape) != 1 or pvalue.size == 0:
            raise ValueError(
                "Only one pattern in 'pattern' may match 'stimulus_pattern'. Either multiple matches or none found."
            )

        # extract pvalues in the set of intervals parameter 'interval'
        # pvals = []
        indices = []
        # t_peak < t_end && t_peak >= t_beg || t_snap_beg<= t_beg && t_snap_end >= t_end
        # note that snap_beg, snap_end and snap_peak are all of the same length -> masks of same size
        for t_beg, t_end in interval:
            idx = np.logical_or(
                np.logical_and(snap_beg <= t_beg, snap_end >= t_end),
                np.logical_and(snap_peak < t_end, snap_peak >= t_beg),
            )

            ix = index.copy()
            ix[pattern_idx, idx] = True
            indices.append(ix)

            # raise ValueError(ix.shape, pvalue.shape)
    return indices


def accuracy(snapshot: np.ndarray, pattern: np.ndarray):
    """
    accuracy between snapshot and pattern

    :param snapshot: population vector within one oscillatory cycle
    :param pattern: original pattern used for learning
    """
    return np.sum(snapshot == pattern)


def log_fac(n: int):
    """
    logarithm of factorial approximated by Stirling's approximation
    ln(n!) ~ n ln(n) - n + 1/2 * ln(2*pi*n)

    :param precision: precision of the approximation (affects number of terms of sterling's series used in approx)
    """
    if n == 0:
        # log(0!) = log(1) = 0
        return 0
    sum_terms = []
    # for truncated series: error is of opposite sign and at most same magnitude as first omitted term
    # -> omitting (truncating the series at) the 4th, 8th or 12th term results in over estimating ie. upper bounding the value (log_fac(n))
    #                                        the 2nd, 6th or 10th term results in underestimating ie. lower bounding the value (log_fac(n))
    # B_{2}...B_{12}:=[0.16666667, 0.0, -0.03333333, 0.0, 0.02380952, 0.0, -0.03333333, 0.0, 0.07575758, 0.0, -0.253113553]
    # -> terms: 2,4,6,8,10,12 are non-trivial as B_{k} != 0.0
    # -> signs of terms (det by (-1)**k * B_{k} as even forall k ~ B_{k}): (k:sign): 2:+,4:-,6:+,8:-, 10:+, 12:-
    for k in [2, 4, 6, 8, 10]:
        # note that bernoulli are real valued
        # sum_terms.append(((-1)**k * float(bernoulli(k))) / (k*(k-1)*n**(k-1)))
        denom = k * (k - 1) * n ** (k - 1)
        if denom > 1e10:
            break
        sum_terms.append(((-1) ** k * float(bernoulli(k))) / denom)
    return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n) + sum(sum_terms)


def log_choose(n: int, k: int):
    """
    log(n choose k) := log(n! / (k! * (n-k)!)) = log(n!) - log(k!) -log((n-k)!)

    :param n: size of set of elements to choose from
    :param k: size subset of elements tb chosen from set paramter n - applies elementwise if array of k values is provided
    """
    if n < k:
        raise ValueError(f"n must be larger than k. Are n {n}, k {k}.")
    # note that effect of log_fac_det over / under estimation depending on sign of truncated term (opposed)
    # does not allow a bound on the binimial coefficient as log_fac values are subtracted here
    return log_fac(n) - log_fac(k) - log_fac(n - k)


def choose(n: int, k: Union[int, np.ndarray]):
    """
    n choose k := n! / (k! * (n-k)!)


    :param n: size of set of elements to choose from
    :param k: size subset of elements tb chosen from set paramter n - applies elementwise if array of k values is provided
    """
    if isinstance(k, np.ndarray):
        if np.any(n < k):
            raise ValueError(f"n must be larger than k. Are n {n}, k {k}.")
        n = np.tile(n, k.size)
    else:
        if n < k:
            raise ValueError(f"n must be larger than k. Are n {n}, k {k}.")
    return factorial(n, exact=True) // (
        factorial(k, exact=True) * factorial(n - k, exact=True)
    )


def p_value_snapshot_same_sparsity(
    similarity: int, sparsity: float, pattern: np.ndarray
):
    """
    p_value of drawing a pattern X = {x_i}, i in [0,l], x_i ~ BER(p=sparsity), xi in {0,1}^l achieving a higher or equal similarity (by chance ~ H0) to parameter similarity
    pmf(p,k)= p^k (1-p)^l-k  - unlike for a binomial distribution the order of trials matters

    probability of an exact match with pattern: p('exact_match') = p^k (1-p)^l-k
    any match that is not an exact match can be expressed starting from the probability of an 'exact_match'  and adjusting for n flips,
    where n-flipped vector is a vector received from pattern by flipping n elements:
    - all combinations of n elements distributed over 1s (p) and 0s (1-p) and chosen from k (#1s) and (l-k) (#0s) respectively
    - adjust k coefficient of probability to compensate for the respective deviations from 'exact_match'
    probability of n-flip: p(n) = sum_{i=max(0,n-(l-k))..min(k,n)} (k choose i) * (l-k choose n-i) * p^(k-i+(n-i)) (1-p)^(l-k+i-(n-i)) (i flips of 1 -> 0 - max number is k, n-i flips of 0->1
        l-k >= n-i <-> i >= n - (l-k), which is a non-trivial constraint if l-k < n)

    probability of achieving a higher or equal similarity s:
    p(N<=l-s) = sum_{n=0..l-s} sum_{i=0..min(k,n)} (k choose i) * (l-k choose n-i) * p^(k-i+(n-i)) (1-p)^(l-k+i-(n-i))

    rewriting the products as sums of logarithms:
    log((k choose i) * (l-k choose n-i) * p^(k-i+(n-i)) (1-p)^(l-k+i-(n-i)))
    ~ log(k choose i) + log(l-k choose n-i) + (k-i+(n-i)) * log(p) + (l-k+i-(n-i)) * log(1-p)

    :param similarity: similarity of a snapshot and pattern pair
    :param sparsity: sparsity parameter used to draw the pattern X = {x_i} i in [0,l], x_i ~ BER(p=sparsity), xi in {0,1}^l
    :param pattern: original pattern
    """
    k = np.sum(pattern)
    l = pattern.size
    s = similarity
    p = sparsity
    return pvalue_snapshot(l, k, s, p)


def pvalue_snapshot_sparsity_missmatch(
    similarity: int,
    sparsity: float,
    pattern: np.ndarray,
    spike_count: int,
    num_cycles: int,
):
    """
    pvalue snapshot sparsity missmatch -  probability of a flip ~ probability of spiking in the snapshot in a given cycle: ps

    p(N<=l-s) = sum_{n=0..l-s} sum_{i=0..min(k,n)} (k choose i) * (l-k choose n-i) * p^(k-i+(n-i)) (1-p)^(l-k+i-(n-i))
    where p = ps
    see pvalue_snapshot, pvalue_snapshost_same_sparsity for derivation

    Note that the maximum number of possible flips n=0..l-s and 1-fips i=0..min(k,n) and their permutations do not change.
    Only the probability with which a snapshot exhibits these flips changes.
    So an exact match is still possible even though it is extremely unlikely.

    On the unchangedness of flips: as the similarity measurement remains the same: s is unchanged, therefore also n=l-s. k is also unchanged as an exact match is
    theoretically possible even for very small ps, even though unlikely, therefore a similarity s of l (s==l) is possible
    and this allows for k 1-flips from this exact match.

    :param similarity: similarity of a snapshot and pattern pair
    :param sparsity: sparsity parameter used to draw the pattern X = {x_i} i in [0,l], x_i ~ BER(p=sparsity), xi in {0,1}^l
    :param pattern: original pattern
    :param spike_count: number of spikes that occurred across the population and simulation time
    :param num_cycles: number of cycles that occurred during simulation (given oscillatory regime)
    """
    l = pattern.size
    p = spike_count / num_cycles / l
    k = np.sum(pattern)

    return pvalue_snapshot(l, k, similarity, p)


def pvalue_snapshot_sparsity_missmatch_single_cycle_count(
    similarity: int,
    sparsity: float,
    pattern: np.ndarray,
    spike_count: int,
):
    """
    pvalue snapshot sparsity missmatch with p snapshot-specific -  probability of a flip ~ probability of spiking in the snapshot in a given cycle: ps

    :param similarity: similarity of a snapshot and pattern pair
    :param sparsity: sparsity parameter used to draw the pattern X = {x_i} i in [0,l], x_i ~ BER(p=sparsity), xi in {0,1}^l
    :param pattern: original pattern
    :param spike_count: number of spikes that occurred across the population and the snapshot
    """
    l = pattern.size
    p = spike_count / l
    k = np.sum(pattern)

    return pvalue_snapshot(l, k, similarity, p)


def pvalue_snapshot(l: int, k: int, s: int, p: float):
    """
    pvalue of snapshot given similarity s with reference pattern ~ probability of observing a more or equally extreme (high) similarity assuming data is distriubted randomly (~ H0)
    note that spiking probability / sparsity can be considered as drawing a RV vector X = [x_i], i in [0,l], x_i ~ BER(p=sparsity), x_i in {0,1}

    P(X >= s) = P(N <= l-s) where X is RV of similarity and N RV of number of flips from exact match
    between pattern and snapshot. To achieve similarity s we must observe exactly l-s flips.
    Any flip is either a flip from 1->0 or from 0->1. Given n flips, observing i, max(0, n-(l-k))<= i <= min(k,n), 1-flips implies also
    observing n-i 0-flips where k: #1s and l-k: #0s. There are k choose i * l-k choose n-i possible
    permutations to observe this.
    Bounds on i the 1-flips/distribution of n flips over 1s and 0s:
    - lower bound: max(0, n-(l-k))<= i ~ dictated by available # of 0s l-k so for n > l-k we will have at least i=n-(l-k) 1s
    - i <= min(k,n) ~ i upper bounded by # 1s in pattern k and number flips n

    :param l: pattern length
    :param k: # 1s in pattern
    :param s: similarity (: accuracy of snapshot given pattern ~ sum(snap==pattern))
    :param p: probability of a flip ~ probability of spiking in the snapshot
    """
    probs = []

    log_p = np.log(p)
    log_ip = np.log(1 - p)
    for n in range(0, l - s + 1):
        # maximum number of 1s to flip is k
        for i in range(max(0, n - (l - k)), min(k, n) + 1):
            probs.append(
                log_choose(k, i)
                + log_choose(l - k, n - i)
                + log_p * (k - i + (n - i))
                + log_ip * (l - k + i - (n - i))
            )

    return np.sum(np.exp(np.array(probs)))


def similarity(snapshot: np.ndarray, pattern: np.ndarray):
    """
    similarity between snapshot and pattern (dot_product)

    :param snapshot: population vector within one oscillatory cycle
    :param pattern: original pattern used for learning
    """
    snapshot = np.array(snapshot, dtype=int)
    pattern = np.array(pattern, dtype=int)
    return snapshot @ pattern


def p_value_snapshot_med(
    similarity: int,
    sparsity: float,
    pattern: np.ndarray,
    num_spikes_snapshot,
    num_cycles,
):
    """
    :param similarity: similarity of a snapshot and pattern pair
    :param sparsity: sparsity parameter used to draw the pattern X = {x_i} i in [0,l], x_i ~ BER(p=sparsity), xi in {0,1}^l
    :param pattern: original pattern
    """
    k = np.sum(pattern)
    l = pattern.size
    d = similarity
    p = sparsity
    p_s = num_spikes_snapshot / num_cycles / l
    probs = []

    log_p = np.log(p)
    log_ip = np.log(1 - p)

    log_ps = np.log(p_s)
    log_ips = np.log(1 - p_s)

    for x in range(d, k + 1):
        for cr in range(d, l - k + d + 1):
            probs.append(
                log_choose(k, d)
                + log_choose(l - k, cr - d)
                + cr * log_ps
                + (l - cr) * log_ips
                + d * log_p
                + (cr - d) * log_ip
            )
    return np.sum(np.exp(np.array(probs)))


def p_value_snapshot_dot_product(
    similarity: int, sparsity: float, pattern: np.ndarray, spike_count, num_cycles
):
    """
    :param similarity: similarity of a snapshot and pattern pair (dot_product)
    :param sparsity: sparsity parameter used to draw the pattern X = {x_i} i in [0,l], x_i ~ BER(p=sparsity), xi in {0,1}^l
    :param pattern: original pattern
    """
    k = np.sum(pattern)
    l = pattern.size
    sim = similarity
    p_s = spike_count / num_cycles / l
    # print("p_s", p_s)
    probs = []

    log_ps = np.log(p_s)
    log_ips = np.log(1.0 - p_s)

    # print("log: ps, ips    ", log_ps, log_ips)

    for d in range(sim, k + 1):
        for cr in range(d, l - k + d + 1):
            probs.append(
                log_choose(k, d)
                + log_choose(l - k, cr - d)
                + cr * log_ps
                + (l - cr) * log_ips
            )
    return np.sum(np.exp(np.array(probs)))


def similarity_threshold(
    sparsity: float,
    pattern: np.ndarray,
    spike_count: int,
    num_cycles: int,
    significance: float = 0.05,
    similarity: np.ndarray = None,
):
    """
    critical threshold, a similarity  value in the given range param similarity, whose p value is the tightest lower bound on the given significance level
    ~ inverse cdf of p_value_snapshot

    :param sparsity: sparsity used to sample pattern from binomial distribution
    :param pattern: pattern for which critical threshold is computed
    :param significance: significance level for which the critical similarity threshold is computed
    :param similarity: range of similarity values from which the critical threshold, a similarity value,
                        whose p value is the tightest lower bound on the given significance level, is selected
                        - assumption: the similarity array
                          is sorted in descending order wrt. the p values of the similarity values as returned by :func:`p_value_snapshot`
                        - default: full search assuming integer interval of similarity values [0, pattern.size - 1]
    :return: critical threshold and corresponding p value which is the tightest lower bound on the significance level (w/in given range, parameter similarity), parameter significance,
             returns (None, None) if no lower bound found ie p_value of similarity[n-1] > significance level
    """

    # no closed form exists for the inverse cdf of binomial distr
    # assumption: similarity array is sorted in descending order wrt p values
    # note p_value_snapshot is a monotonically decreasing function as p values decrease with similarity
    #  - as incr sim means decr number of flips as nflips = size - sim
    #     and we are summing over flips from 0 to size-sim
    #  -> all sum terms contained in sim = x - 1 (nflips = size - x + 1) are also contained in sim = x (nflips = size-x) and additionally the sum term
    #         for one additional flip (flip = size - x) which is > 0 . Hence monotonically decreasing.
    # -> guarantees unique critical threshold (tightest lower bound) if it exists
    # -> we can use binomial search variant where we iterate over input array and never materialize
    #    the entire result array (as input order reversed in result array)

    # so binomial search variant is used to compute an upper bound on the significance level

    if isinstance(similarity, type(None)):
        similarity = np.arange(0, pattern.size)

    ubound = (None, None)

    p = spike_count / num_cycles / pattern.size

    if not (p > 0 or p < 1):
        return ubound

    l = 0
    r = pattern.size - 1
    while l <= r:
        m = (l + r) // 2
        sim = similarity[m]
        # print(m, sim, sparsity, pattern)
        p_m = pvalue_snapshot(l=pattern.size, k=np.sum(pattern), s=sim, p=p)
        if p_m < significance:
            ubound = (sim, p_m)
            r = m - 1
        elif p_m > significance:
            l = m + 1
        else:  # ==
            ubound = (sim, p_m)
            break
    return ubound


def similarity_conductance_scaling(pattern: np.ndarray):
    """
    similarity rule for conductance scaling ~ pair-wise similarity btw two neurons (ie. per synapse) averaged across patterns

    :param pattern: patterns (N x pop_size) from which the weights are computed
    """
    # (pop_size x N) @ (N X pop_size) = pop_size x pop_size
    return sim_vec(pattern.T)


def sim_vec(matrix: np.ndarray):
    """
    similarity of row vectors

    r_ij = dot product of (row) vectors i,j of length l divided by l,
    where matrix is of dimensions (n,l)
    """
    return matrix @ matrix.T / matrix.shape[1]


def compute_conductance_scaling(patterns: np.ndarray, sparsity: float):
    """
    compute the scaling factor of the conductance  according to Battaglia, Treves 1998
    (https://pubmed.ncbi.nlm.nih.gov/9472489/)
    original process: i) g_ij := 0 ii) for each pattern do:
        a) delta_g_ij = g_EE / C_EE * (n_i^p / sparsity - 1) (n_j^p / sparsity - 1) b) g_ij = max(0, g_ij + delta_g_ij)
    where g_ij is the conductance of synapse from neuron with index i to j

    Here scaling factor s is computed:
    - g = g_EE / C_EE * s
    - process: i) s_ij := 0 ii) for each pattern do:
        a) delta_s_ij = (n_i^p / sparsity - 1) (n_j^p / sparsity - 1) b) s_ij = max(0, s_ij + delta_s_ij)
    (clipping equivalent to original process as g_EE/C_EE is a positive constant term therefore
        crossing of 0 (clipping) remains unchanged)


    :param patterns: patterns tb used in compution shape: (p,size) where size is the size of the pattern
                    (= size of E population) and p is the number of patterns
    :param sparsity: sparsity of the patterns
    :return: scaling factor s for conductances (shape: (size,size))
    """

    if sparsity < 0.0 or sparsity > 1.0:
        raise ValueError(f"sparsity must be in [0,1]. Is {sparsity}.")

    size = patterns.shape[1]
    s = np.zeros(size * size).reshape(size, size)

    for p in range(patterns.shape[0]):
        pattern = patterns[p] / sparsity - 1
        delta_s = np.outer(pattern, pattern)
        s = np.maximum(0, delta_s + s)

    return s


def compute_conductance_scaling_single_clip(patterns: np.ndarray, sparsity: float):
    """
    compute the scaling factor of the conductance by summing over patterns and clipping the result

    Here scaling factor s is computed:
    - g = g_EE / C_EE * s
    - process: i) for each pattern compute s^p_ij = (n_i^p / sparsity - 1) (n_j^p / sparsity - 1)
              ii) s_ij = sum_p s^p_ij
              iii) s_ij = max(0, s_ij)


    :param patterns: patterns tb used in compution shape: (p,size) where size is the size of the pattern
                    (= size of E population) and p is the number of patterns
    :param sparsity: sparsity of the patterns
    :return: scaling factor s for conductances (shape: (size,size))
    """

    if sparsity < 0.0 or sparsity > 1.0:
        raise ValueError(f"sparsity must be in [0,1]. Is {sparsity}.")

    # patterns: shape (num_p, size) one row ~ a pattern; ij,ib->jb: i) create outer product for each row ii) sum over rows
    # (np.sum(np.vstack([np.outer(p,p).reshape(1,p.size, p.size) for p in patterns / sparsity - 1]), axis=0))
    pat = patterns / sparsity - 1
    s = np.einsum("ij,ib->jb", pat, pat)
    return np.maximum(0, s)


def compute_conductance_scaling_unclipped(patterns: np.ndarray, sparsity: float):
    """
    compute the scaling factor of the conductance by summing over patterns

    Here scaling factor s is computed:
    - g = g_EE / C_EE * s
    - process: i) for each pattern compute s^p_ij = (n_i^p / sparsity - 1) (n_j^p / sparsity - 1)
              ii) s_ij = sum_p s^p_ij


    :param patterns: patterns tb used in compution shape: (p,size) where size is the size of the pattern
                    (= size of E population) and p is the number of patterns
    :param sparsity: sparsity of the patterns
    :return: scaling factor s for conductances (shape: (size,size))
    """

    if sparsity < 0.0 or sparsity > 1.0:
        raise ValueError(f"sparsity must be in [0,1]. Is {sparsity}.")

    pat = patterns / sparsity - 1
    return np.einsum("ij,ib->jb", pat, pat)


def normalize(matrix: np.ndarray, frm: float = None, to: float = None):
    """
    normalize (here squash) all values in matrix to [0,1] and if specified rescale to [frm,to]

    :param matrix: matrix tb normalized
    :param frm: lower bound of interval to which matrix is tb rescaled (requires setting to)
    :param to: upper bound of interval to which matrix is tb rescaled (requires setting frm)
    :return: matrix normalized to [0,1] or [frm,to] if specified
    """
    if (frm == None or to == None) and frm != to:
        raise ValueError("frm and to must either both be specified or neither.")
    mn = np.min(matrix)
    mx = np.max(matrix)
    norm_m = (matrix - mn) / (mx - mn)
    return norm_m * (to - frm) + frm if frm != None and to != None else norm_m


def z_score(matrix: np.ndarray):
    """
    compute z score: z := (x - mu) / sigma
    where mu = mean(matrix), sigma = std(matrix) (over all values in matrix),
    for all values x in matrix

    :param matrix: matrix tb normalized
    :return: z_score of the matrix
    """
    return (matrix - np.mean(matrix)) / np.std(matrix)


def extract_snapshot_masks(
    pop_rate: np.ndarray, t_start: float, t_end: float, dt: float
):
    """
    extract snapshot masks of oscillation cycles,
    where one snapshot mask is a boolean mask for a specific cylce (trough-to-trough)
    with value at index i True iff neuron i spiked in the respective cycle

    :param pop_rate: population rate from which snapshot masks are generated for interval [t_start, t_end] - resolution must be the simulation timestep, parameter dt
    :param t_start: start time for analysis and extraction
    :param t_end: end time for analysis and extraction
    :param dt: time step of the simulation and resolution of snapshot masks
    :return: snapshot masks of oscillation cycles as csr matrix (sparse): (C,T), where number of cycles C = troughs.size -1 and time bins T = (t_end-t_start)//dt + 1
    """

    # deal with 2 cases: when restricted to interval [t_start, t_end] upper bound is included vs not the case for no upperbound
    # - brian2 simulation ends with timestep t_end - dt
    time = np.arange(t_start, t_end + dt, dt)[: pop_rate.size]

    trough_idx = analysis.detect_peaks(-pop_rate, dt)
    inter_trough_intervals = (
        time[trough_idx][1:] - time[trough_idx][:-1]
        if len(trough_idx) > 1
        else np.array([])
    )

    vector_masks = sp.lil_matrix(
        (inter_trough_intervals.size, pop_rate.size), dtype=bool
    )  # np.zeros((inter_trough_intervals.size, pop_rate.size), dtype=bool)
    for i in range(inter_trough_intervals.size):
        vector_masks[i] = np.logical_and(
            time >= time[trough_idx][i],
            time <= time[trough_idx][i] + inter_trough_intervals[i],
        )
    return vector_masks.tocsr()


def extract_snapshots(
    spike_train: Dict[str, np.ndarray],
    pop_size: int,
    pop_rate: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float,
):
    """
    :spike_train: spike_train per neuron [ms]
    :param pop_size: size of the population - assumes neuron indices in [0, pop_size) whose str representations are keys of spike_train
    :param t_start: start time for snapshot extraction [ms]
    :param t_end: end time for snapshot extraction [ms]
    :param dt: step size of simulation
    :return: snapshots (C x pop_size), where C is the number of cylces
    """

    # pop_rate.size ~ number of time steps
    spikes = sp.lil_matrix((pop_rate.size, pop_size))
    for n in spike_train.keys():
        spike_tr = spike_train[n][
            np.logical_and(spike_train[n] >= t_start, spike_train[n] <= t_end)
        ]
        spike_idx = np.array((spike_tr - t_start) / dt, dtype=int)

        # assumes neuron indices in [0, pop_size)
        spikes[spike_idx, int(n)] = 1

    # spikes: csr matrix : T x pop_size, ie a row is a boolean vector indicating spikes within the respective time bin
    spikes = sp.csr_matrix(spikes)

    # vector masks: csr matrix: C x T -> T for one cycle
    # spikes T x pop_size
    # np.sum(spikes[T], axis=0) == T @ spikes
    vector_masks = extract_snapshot_masks(pop_rate, t_start, t_end, dt)

    # csr mat mult: snapshot C x pop_size - indicating spike within cycle C for each neuron
    # (cast to bool also deals with multiple spikes of a neuron per cycle)
    snapshots = np.asarray((vector_masks @ spikes).toarray(), dtype=bool)

    return snapshots
