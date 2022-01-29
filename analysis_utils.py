from operator import ne
import numpy as np


def from_plateaus_compute_falling_and_rising_edges(
    x: np.ndarray, plateau_idx: np.ndarray
):
    """
    compute the rising and falling edges of the plateau
    helper for :func:`detect_peaks`
    """
    plats = []
    cur_plat = []
    for p in plateau_idx:
        if len(cur_plat) == 0:
            cur_plat.append(p)
        elif p - cur_plat[-1] != 1:
            start, end = cur_plat[0], cur_plat[-1]
            if start > 0 and end < len(x) - 1:
                plats.append((start - 1, end + 1))
            cur_plat = [p]
        else:
            cur_plat.append(p)
    if len(cur_plat) > 0:
        start, end = cur_plat[0], cur_plat[-1]
        if start > 0 and end < len(x) - 1:
            plats.append((start - 1, end + 1))
    return plats


def rectify_plats(plats, min_mask: np.ndarray, max_mask: np.ndarray):
    """
    replace the falling and rising edge of all peaks/troughs with plateaus with the middle index
    helper for :func:`detect_peaks`
    """
    for start, end in plats:
        if min_mask[start] and min_mask[end]:
            min_mask[start] = False
            min_mask[end] = False
            # set middle true
            min_mask[(end - start) // 2 + start] = True
        elif max_mask[start] and max_mask[end]:
            max_mask[start] = False
            max_mask[end] = False
            # set middle true
            max_mask[(end - start) // 2 + start] = True

        # remove saddle points (with plateau)

        elif min_mask[start] and max_mask[end]:
            min_mask[start] = False
            max_mask[end] = False
        elif max_mask[start] and min_mask[end]:
            max_mask[start] = False
            min_mask[end] = False
    return min_mask, max_mask


def rectify_subsequent_falling_rising_edge_without_plats(x, min_idx, max_idx):
    """
    replace the falling and rising edge (ie. extremum is not a single point) of all peaks/troughs without plateaus by rising edge
    and remove the saddle point indices in min_idx, max_idx that have no plateau
    note that the highest number of subsequent values is 2 as otw we would have a plateau of length >= 1
    helper for :func:`detect_peaks`
    """
    # replace the falling and rising edge (ie. extremum is not a single point) of all peaks/troughs without plateaus by rising edge
    dupes_idx = np.array(
        [
            i + 1
            for i, (pred, cur) in enumerate(zip(min_idx[:-1], min_idx[1:]))
            if pred == cur - 1
        ]
    )
    if len(dupes_idx) > 0:
        min_idx = np.delete(min_idx, dupes_idx)

    dupes_idx = np.array(
        [
            i + 1
            for i, (pred, cur) in enumerate(zip(max_idx[:-1], max_idx[1:]))
            if pred == cur - 1
        ]
    )
    if len(dupes_idx):
        max_idx = np.delete(max_idx, dupes_idx)

    # Remove saddle points

    # all min_idx and max_idx are now alternating ie if merged and sorted one would map to all i % 2 == 0 indices and the other ==1
    # we need to remove the saddle point that that do not have a plateau - these have one edge (raising/falling) in min_idx and the other in max_idx
    #  as we are alternating between min_idx and max_idx, subsequent indices in the joint array must be a peak/trough and trough/peak, respectively
    #    if the values of these indices are subsequent as well this means there is a peak/trough following a trough/peak directly in the vector x
    #    which in turn means we have found a saddle point iff value of x at both indices is equal
    #    note: the falling/raising edges of peaks and troughs are inputs to this function and referred to in above commet simply as peak and trough
    #          without distinction

    extrema = np.sort(np.hstack((min_idx, max_idx)))
    saddle_idx = []
    for i, (pred, cur) in enumerate(zip(extrema[:-1], extrema[1:])):
        if pred == cur - 1 and x[pred] == x[cur]:
            # whichever starts the alternation min_idx or max_idx receives the first index and the other the second
            saddle_idx.append([int(np.ceil(i / 2)), int(np.floor(i / 2))])

    if len(saddle_idx) > 0:
        # obv only the case if both len(min_idx) > 0 and len(max_idx) > 0 as otw len(extrema) <= 1
        first_is_max = max_idx[0] < min_idx[0]

        saddle_idx = np.array(saddle_idx).T
        saddle_min, saddle_max = (
            (saddle_idx[0], saddle_idx[1])
            if not first_is_max
            else (saddle_idx[1], saddle_idx[0])
        )
        min_idx = np.delete(min_idx, saddle_min)
        max_idx = np.delete(max_idx, saddle_max)

    return min_idx, max_idx


def detect_peaks(x: np.ndarray):
    """
    detect all peaks and troughs (ignore saddle points)

    There are 3 types of extrema: peak, trough, sadddle.
    Peak and trough must be rectified if they have a plateau or a falling and rising edge (ie not a single point / the extremum is two points of equal value)
    and the indices involved in saddle points deleted: either min_idx and max_idx that are subsequent or connected by plateau
    (see rectify_plats and rectify_subsequent_falling_rising_edge_without_plats).
    """

    successor_diff = (x[0:-1] - x[1:])[1:]
    #  diff to predecessor evaluated for x[1]-x[n] + rectification (remove last)
    predecessor_diff = (x[1:] - x[:-1])[:-1]

    # this may select multiple points for peaks/troughs and saddlepoints
    # ('rising and falling edges' in case the extremum does not reside in a single point)
    #   for plateaus around extrema
    #   (eg. saddle point with multiple equal values - discretization)
    mins = np.logical_or(
        np.logical_and(successor_diff < 0, predecessor_diff <= 0),
        np.logical_and(successor_diff <= 0, predecessor_diff < 0),
    )
    maxs = np.logical_or(
        np.logical_and(successor_diff > 0, predecessor_diff >= 0),
        np.logical_and(successor_diff >= 0, predecessor_diff > 0),
    )
    plateaus = np.logical_and(successor_diff == 0, predecessor_diff == 0)

    # rectification for shortening vectors in pred, succ comparison
    mins = np.hstack((np.array([False]), mins, np.array([False])))
    maxs = np.hstack((np.array([False]), maxs, np.array([False])))
    plateaus = np.hstack((np.array([False]), plateaus, np.array([False])))

    plateaus = plateaus.nonzero()[0]

    plats = from_plateaus_compute_falling_and_rising_edges(x, plateaus)

    mins_rect, maxs_rect = rectify_plats(plats, mins, maxs)
    mins_rect = mins_rect.nonzero()[0]
    maxs_rect = maxs_rect.nonzero()[0]
    mins_rect, maxs_rect = rectify_subsequent_falling_rising_edge_without_plats(
        x, mins_rect, maxs_rect
    )

    return mins_rect, maxs_rect


def detect_symmetric_peaks(x, delta):
    """
    detect peaks applying a threshold symmetrically in time (antero- and retrograde) according to an OR rule
    ie. peaks/troughs are detected if the difference in amplitude to the previous trough/peak or the subsequent trough/peak
    larger or equal to the threshold
    """

    mins, maxs = detect_peaks(x)

    # otw constant fct - treated here as having no extrema
    if len(maxs) != 0 or len(mins) != 0:

        # note that we detected all local min and max (no thresholds etc.) via detect_peaks(x) so per def. we know that they must alternate
        first_is_max = (
            maxs[0] < mins[0] if len(maxs) > 0 and len(mins) > 0 else len(maxs) > 0
        )
        next_is_max = first_is_max
        last_was_max = not next_is_max

        extrema = np.sort(np.hstack((mins, maxs)))
        extrema_delta = []
        twix_peak_tmp = None
        twix_trough_tmp = None

        for i, to in enumerate(extrema):

            # note that next_is_max == True <-> twix_trough_tmp != None  (with exception of the very first iteration)
            # bc. next_is_max == True when the previous extremum was a min which means either the diff was >= delta and min was added
            #   and thereby twix_trough_tmp = that min or it was smaller delta and then twix_trough_tmp is updated to a value that is
            #   != None (see outermost else branch)
            # (twix_peak_tmp analgously)
            #  next_is_max is still needed for cases in which both twix_trough_tmp and twix_peak_tmp are != None eg . multiple peaks and
            #   troughs with differences smaller delta
            if (
                next_is_max
                and np.abs(x[to] - x[twix_trough_tmp if i != 0 else 0]) >= delta
                or not next_is_max
                and np.abs(x[to] - x[twix_peak_tmp if i != 0 else 0]) >= delta
            ):

                if next_is_max:
                    # inside below if-branch twix_trough_tmp is guaranteed tb != None
                    # see comment of the outer most if-branch in for loop for explanation
                    # additionally for the first iteration next_is_max != last_was_max (per init.) and therefore if-branch is never
                    #   entered on first iteration
                    if last_was_max:
                        # append trough if most recently added extrema was peak and trough did not reach delta threshold to previous peak as current peak has reached delta threshold wrt to this trough
                        #   or if no extrema have yet crossed the delta threshold yet a trough preceeded this peak
                        extrema_delta.append(twix_trough_tmp)

                    extrema_delta.append(to)
                    twix_peak_tmp = to
                    twix_trough_tmp = None
                    last_was_max = True

                else:
                    # see if-branch
                    if not last_was_max:
                        # see if-branch
                        extrema_delta.append(twix_peak_tmp)
                    extrema_delta.append(to)
                    twix_trough_tmp = to
                    twix_peak_tmp = None
                    last_was_max = False

            else:
                # keep the most extreme local extrema (peak/trough) as long as delta is not reached,
                #        as a reference point for the other extremum type (trough/peak)
                # note that if a peak is detected twix_trough_tmp is set to None
                #      and  if a trough is detected twix_peak_tmp is set to None
                if next_is_max:
                    if isinstance(twix_peak_tmp, type(None)):
                        twix_peak_tmp = to
                    elif x[twix_peak_tmp] < x[to]:
                        twix_peak_tmp = to
                else:
                    if isinstance(twix_trough_tmp, type(None)):
                        twix_trough_tmp = to
                    elif x[twix_trough_tmp] > x[to]:
                        twix_trough_tmp = to

            # flip
            next_is_max = not next_is_max

        extrema_delta = np.array(extrema_delta)
        mins = extrema_delta[::2] if not first_is_max else extrema_delta[1::2]
        maxs = extrema_delta[::2] if first_is_max else extrema_delta[1::2]

    return mins, maxs
