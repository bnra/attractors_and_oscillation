from turtle import xcor
import matplotlib
from typing import Dict, Iterable, Iterator, Union, List, Tuple
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns

import persistence
from utils import compute_time_interval, restrict_to_interval


def discrete_palette(n, color="husl"):
    """
    discrete cirular color palette

    :param n: number of colors tb generated
    :param color: color tb used for :func:'sns.color_palette'
    """
    cmap = sns.color_palette(color, as_cmap=True)
    for i in np.linspace(0, 1, n):
        yield cmap(i)


def sequential_palette(base_color: str, n: int):
    """
    create a generator over sequential colors of a primary hue, base_color

    :param base_color: base color for colormap
    :param n: length of the iterator
    """
    cols = ["Blues", "Greens", "Greys", "Oranges", "Purples", "Reds"]
    if base_color not in cols:
        raise ValueError(
            f"No such base_color: supported colors are {cols}. Is {base_color}."
        )
    cmap = sns.color_palette(base_color, as_cmap=True).reversed()
    return color_palette(cmap, n)


def color_palette(cmap: matplotlib.colors.LinearSegmentedColormap, n: int):
    """
    generator over the color map of length n

    :param cmap: linear segmented colormap
    :param n: length of the iterator
    """
    for i in np.linspace(0.25, 0.75, n):
        yield cmap(i)


color_its = {
    "E": lambda n: sequential_palette("Reds", n),
    "I": lambda n: sequential_palette("Blues", n),
}

colors = {
    "E": next(sequential_palette("Reds", 10)),
    "I": next(sequential_palette("Blues", 10)),
}


def subdivide_subplot(fig: plt.Figure, ax: plt.Axes.axes, rows: int, cols: int):
    """
    subdivide axes / subplot into a grid of axes

    :param fig: figure to which parameter ax belongs
    :param ax: axes / subplot which is tb subdivided
    :param rows: number of rows of the grid into which the parameter ax is subdivided
    :param cols: number of cols of the grid into which the parameter ax is subdivided
    """
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )
    axes = []
    gs = gridspec.GridSpecFromSubplotSpec(
        rows, cols, subplot_spec=ax, wspace=0.0, hspace=0.0
    )
    for i in range(rows):
        for j in range(cols):
            axx = plt.Subplot(fig, gs[i, j])
            fig.add_subplot(axx)
            axes.append(axx)

    return np.array(axes).reshape(rows, cols)


def psth_snap(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    snapshots: np.ndarray,
    snap_times: np.ndarray,
    stimulus_pattern: np.ndarray,
    num_presentations: int,
    inter_presentation_interval: float,
    stimulus_length: float,
    num_bins: int = 10,
):
    """
    Peristimulus Time Histogram for snapshot activity over stimulus presentation cycles

    :param fig: figure instance
    :param ax: axis instance
    :param snapshots: snapshots (C, pattern_length), where C is number snapshots
    :param snap_times: snapshot firing times resolved relative to stimulus [ms]
    :param num_presentations: number of stimulus presentations
    :param inter_presentation_interval: interval between two stimulus presentation onsets or ceasures equivalently [ms]
    :param stimulus_length: length of stimulus presentation [ms]
    :param num_bins: number of bins
    """
    delta = inter_presentation_interval / num_bins

    counts = np.zeros((snapshots.shape[1], num_bins))

    for i in range(num_bins):
        counts[:, i] = np.sum(
            snapshots.T[
                :, np.logical_and(snap_times >= i * delta, snap_times < (i + 1) * delta)
            ],
            axis=1,
        )
    rate = counts / (num_presentations * delta)

    # print(rate.shape)

    # reorder - ones on top
    ones_idx = stimulus_pattern == True
    zeros_idx = ones_idx == False
    rate_reordered = np.zeros_like(rate)
    # imshow shows pixels from top left to bottom right - so (0,0) is top left and not on the bottokm
    rate_reordered[0 : np.sum(ones_idx)] = rate[ones_idx]
    rate_reordered[np.sum(ones_idx) :] = rate[zeros_idx]

    # plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)

    im = ax.imshow(
        rate_reordered,
        aspect="auto",
        cmap="jet",
        extent=[0.0, inter_presentation_interval, 0, snapshots.shape[1]],
    )
    ax.vlines(stimulus_length, 0, snapshots.shape[1], color="black")
    ax.set_ylabel(
        f"neuron ids (1 cells top {np.sum(ones_idx)})"  
    )
    ax.set_xlabel(
        f"time relative to stimulus onset [ms] (stimulus end at {stimulus_length:.2f} ms)"
    )
    fig.colorbar(
        im,
        cax=cax,
        orientation="vertical",
        label="snapshot rate", 
    )


def psth(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    spike_train_times: np.ndarray,
    spike_train_ids: np.ndarray,
    stimulus_pattern: np.ndarray,
    num_presentations: int,
    inter_presentation_interval: float,
    stimulus_length: float,
    num_bins: int = 10,
):
    """
    Peristimulus Time Histogram for snapshot activity over stimulus presentation cycles
    [stimulus_onset - inter_presentation_interval + stimulus_length, stimulus_onset + inter_presentation_interval]

    :param fig: figure instance
    :param ax: axis instance
    :param spike_times: timings of spikes [ms] (S,), where S is number spike events
    :param spike_ids: ids of neuron for which a spike event is evoked (S,), where S is number spike events
    :param num_presentations: number of stimulus presentations
    :param inter_presentation_interval: interval between two stimulus presentation onsets or ceasures equivalently [ms]
    :param stimulus_length: length of stimulus presentation [ms]
    :param num_bins: number of bins
    """

    spikes = {}
    # np.unique will sort unique values ascendingly
    for idx in np.unique(spike_train_ids):
        spikes[idx] = spike_train_times[spike_train_ids == idx]

    delta = (2 * inter_presentation_interval - stimulus_length) / num_bins

    # (N, B) where N is number of neurons and B is num_bins
    counts = np.zeros((stimulus_pattern.size, num_bins))

    ids = sorted(list(spikes.keys()))
    for j in ids:
        counts[int(j)] = np.array(
            [
                np.sum(
                    np.logical_and(
                        spikes[j]
                        >= i * delta - (inter_presentation_interval - stimulus_length),
                        spikes[j]
                        < (i + 1) * delta
                        - (inter_presentation_interval - stimulus_length),
                    )
                )
                for i in range(num_bins)
            ]
        )

    # rate in Hz: * 1000 as delta is in ms on log_scale + epsilon shift

    rate = np.log(counts / (num_presentations * delta) * 1000 + 1e-10)

    # reorder - ones on top
    ones_idx = stimulus_pattern == True
    zeros_idx = ones_idx == False
    rate_reordered = np.zeros_like(rate)
    # imshow shows pixels from top left to bottom right - so (0,0) is top left and not on the bottokm
    rate_reordered[0 : np.sum(ones_idx)] = rate[ones_idx]
    rate_reordered[np.sum(ones_idx) :] = rate[zeros_idx]

    # print(np.arange(0, inter_presentation_interval+1e-5, delta).shape, (counts / (num_presentations * delta)).shape)
    # plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)

    im = ax.imshow(
        rate_reordered,
        aspect="auto",
        cmap="jet",
        extent=[
            stimulus_length - inter_presentation_interval,
            inter_presentation_interval,
            0,
            rate.shape[0],
        ],
    )
    ax.vlines(0, 0, rate.shape[0], color="black")
    ax.vlines(stimulus_length, 0, rate.shape[0], color="black")
    ax.set_ylabel(
        f"neuron ids (1 cells top {np.sum(ones_idx)})"  # [{snapshots.shape[1]-np.sum(ones_idx)},{snapshots.shape[1]-1}])"
    )
    ax.set_xlabel(
        f"time relative to stimulus onset [ms] (stimulus end at {stimulus_length:.2f} ms)"
    )
    vmin = np.min(rate_reordered)
    vmax = np.max(rate_reordered)
    cb = fig.colorbar(
        im,
        cax=cax,
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        orientation="vertical",
        label="log cell rate [Hz]",  # (active in num snapshots per presentation and bin size)",
    )

    # cb.set_clim(np.min(rate_reordered), np.max(rate_reordered))


def plot_spike_train_interval(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    spike_train_ids: np.ndarray,
    spike_train_times: np.ndarray,
    lbound: float,
    ubound: float,
    pop_rate: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float,
):
    """
    plot spikes of a specific time interval [lbound, ubound]

    :param fig: figure instance
    :param ax: axis instance
    :param spike_train_ids: neuron ids of spikes
    :param spike_train_times: spike times of spikes [ms]
    :param lbound: lower bound of time interval [ms]
    :param ubound: upper bound of time interval [ms]
    :param pop_rate: population rate
    :param t_start: start of simulation/analysis
    :param t_end: end of simulation/analysis
    :param dt: time step of simulation
    """
    idx = np.logical_and(spike_train_times >= lbound, spike_train_times <= ubound)
    spikes = spike_train_times[idx]
    ids = spike_train_ids[idx]

    time = np.arange(t_start, t_end, dt)

    pop_rate_idx = np.logical_and(time >= lbound, time <= ubound)

    # 2 cases: normal end pop_rate [0,t) where t is simulation time vs restrictiction to upperbound [0, t_end] -> 1 more value
    # simply curtail to length of [0,t) first
    pr = pop_rate[: time.size][pop_rate_idx]
    tm = time[pop_rate_idx]


    ax.scatter(spikes, ids, marker=".", s=(300.0 / fig.dpi) ** 2)
    ax.set_xlabel("spike time [ms]")

    prax = ax.twinx()
    prax.set_ylabel("pop rate [Hz]")
    prax.plot(tm, pr, alpha=0.5, color="orange")

    # done to rescale prax equivalently to ax
    divider = make_axes_locatable(prax)
    spaceax = divider.append_axes("left", size="10%", pad=0.0)
    spaceax.remove()

    divider = make_axes_locatable(ax)
    crax = divider.append_axes("left", size="10%", pad=0.0)

    ids_rate = np.unique(ids)

    rate = [np.sum(ids == i) / ((ubound - lbound) / 1000) for i in ids_rate]
    crax.scatter(rate, ids_rate, marker=".", s=(300.0 / fig.dpi) ** 2)
    crax.set_xlabel("cell rate [Hz]")  # (interval)
    crax.set_ylabel("neuron ids (1 cells from top)")


def plot_spike_train_presentation_cycle(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    spike_train_ids: np.ndarray,
    spike_train_times: np.ndarray,
    stimulus_onset: float,
    inter_presentation_interval: float,
    stimulus_length: float,
    pop_rate: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float,
    stimulus_pattern: np.ndarray = None,
):
    """
    plot spikes of a specific presentation cycle [stimulus_onset - inter_presentation_interval, stimulus_onset + inter_presentation_interval]

    :param fig: figure instance
    :param ax: axis instance
    :param spike_train_ids: neuron ids of spikes
    :param spike_train_times: spike times of spikes [ms]
    :param stimulus_pattern: one of the patterns embedded in the weight matrix and whose perturbation is used as a stimulus (pop_length,)
    :param stimulus_onset: onset of stimulus presentation [ms]
    :param inter_presentation_interval: interval between two stimulus presentation onsets [ms]
    :param stimulus_length: length of stimulus [ms]
    :param pop_rate: population rate
    :param t_start: start time of pop_rate
    :param t_end: end time of pop_rate simulation/analysis
    :param dt: resolution of pop_rate
    """
    # reorder - note that ones are at the top and zeros at the bottom and ids are sorted within each group

    idx_ordered = np.argsort(spike_train_ids)
    idx_reordered = np.zeros_like(idx_ordered, dtype=int)

    spike_train_ids_ordered = spike_train_ids[idx_ordered]

    if not isinstance(stimulus_pattern, type(None)):

        ones_idx = np.nonzero(stimulus_pattern == True)[0]
        zero_idx = np.nonzero(stimulus_pattern == False)[0]

        zero_spikes = np.hstack(
            [np.nonzero(spike_train_ids_ordered == e)[0] for e in zero_idx]
        )

        idx_reordered[0 : zero_spikes.size] = zero_spikes

        idx_reordered[zero_spikes.size :] = np.hstack(
            [np.nonzero(spike_train_ids_ordered == e)[0] for e in ones_idx]
        )

        spike_train_ids_reordered = spike_train_ids[idx_ordered][idx_reordered]

        reordered_ids = np.zeros_like(spike_train_ids)

        offset = 0
        # np.unique will sort unique values ascendingly
        for idx in np.unique(spike_train_ids):
            # exploiting that spike_train_ids_reordered is sorted within each group (0s and 1s) and therefore all entries with same id are contiguous
            size_next_id_slice = np.sum(
                spike_train_ids_reordered == spike_train_ids_reordered[offset]
            )
            reordered_ids[offset : offset + size_next_id_slice] = (
                np.ones(size_next_id_slice) * idx
            )
            offset += size_next_id_slice

        reordered_spike_times = spike_train_times[idx_ordered][idx_reordered]
    else:
        reordered_ids = spike_train_ids_ordered
        reordered_spike_times = spike_train_times[idx_ordered]

    # plot

    plot_spike_train_interval(
        fig,
        ax,
        reordered_ids,
        reordered_spike_times,
        stimulus_onset - inter_presentation_interval + stimulus_length,
        stimulus_onset + inter_presentation_interval,
        pop_rate,
        t_start,
        t_end,
        dt,
    )
    ax.hlines(-10, stimulus_onset, stimulus_onset + stimulus_length)


def plot_similarity_top_snaps(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    pop_name: str,
    title: str,
    rows: int,
    cols: int,
    similarity_threshold: int,
    similarity_distribution: np.ndarray,
):
    """
    plot top snapshots for which the threshold is exceeded for each pattern

    :param fig: figure instance
    :param ax: axis instance
    :param rows: number of rows into which the axes/subplot is subdivided
    :param cols: number of columns into which the axes/subplot is subdivided
    :param similarity_threshold: similarity threshold corresponding to the tightest lower bound on the significance level for all patterns - assumes patterns have fixed (and equal) # 1s
    :param similarity_distribution: similarities between patterns and snapshots (N x C) where N is the number of patterns and C the number of cycles
    """
    num_patterns = similarity_distribution.shape[0]
    if not rows * cols >= num_patterns:
        raise ValueError(
            "product of parameters 'rows' and 'cols' must be >= N, number of patterns in 'similarity_distribution' and 'similarity_threshold'"
        )

    axes = subdivide_subplot(fig, ax, rows, cols)

    # print(similarity_threshold.shape, similarity_distribution.shape)

    for i in range(num_patterns):
        axes[i % rows, i // cols].tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )

        sims = similarity_distribution[i]
        top_idx = np.where(sims >= similarity_threshold)[0]

        if top_idx.size > 0:
            for j in range(top_idx.size):
                axes[i % rows, i // rows].scatter(
                    top_idx[j : j + 1],
                    sims[top_idx][j : j + 1],
                    label=f"({top_idx[j]},{sims[top_idx][j]})",
                )
            xmn = np.min(top_idx) if top_idx.size > 1 else top_idx[0] - 0.5
            xmx = np.max(top_idx) if top_idx.size > 1 else top_idx[0] + 0.5
            axes[i % rows, i // rows].hlines(
                similarity_threshold,
                xmin=xmn,
                xmax=xmx,
                label=f"{similarity_threshold} (P {i})",
                color="red",
            )
            axes[i % rows, i // rows].legend()
            # axes[i % rows, i // rows].text(
            #     np.min(top_idx),
            #     similarity_threshold,
            #     f"{similarity_threshold:.0f}",
            #     horizontalalignment="center",
            #     verticalalignment="center",
            #     fontsize="xx-small",
            # )
            # for j in range(top_idx.size):
            #     axes[i % rows, i // rows].text(
            #         top_idx[j],
            #         sims[top_idx][j],
            #         f"({top_idx[j]},{sims[top_idx][j]})",
            #         horizontalalignment="center",
            #         verticalalignment="center",
            #         fontsize="xx-small",
            #     )

        # for j, v in enumerate(hplot[0]):
        #     bin_val = (hplot[1][j] + hplot[1][j + 1]) / 2
        #     axes[i % rows, i // rows].text(
        #         bin_val,
        #         v + 1,
        #         f"({bin_val:.2f},{v:.2f})",
        #         horizontalalignment="center",
        #         verticalalignment="center",
        #     )
        # axes[i % rows, i // rows].legend()

    ax.set_title(f"top snaps (snap,sim) {pop_name} {title}")


def plot_similarity_distributions(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    pop_name: str,
    title: str,
    rows: int,
    cols: int,
    similarity_threshold: int,
    similarity_distribution: np.ndarray,
):
    """
    plot similarity distributions with a single (same) threshold

    :param fig: figure instance
    :param ax: axis instance
    :param rows: number of rows into which the axes/subplot is subdivided
    :param cols: number of columns into which the axes/subplot is subdivided
    :param similarity_threshold: similarity threshold corresponding to the tightest lower bound on the significance level for all patterns - assumes patterns have fixed (and equal) # 1s
    :param similarity_distribution: similarities between patterns and snapshots (N x C) where N is the number of patterns and C the number of cycles
    """
    num_patterns = similarity_distribution.shape[0]
    if not rows * cols >= num_patterns:
        raise ValueError(
            "product of parameters 'rows' and 'cols' must be >= N, number of patterns in 'similarity_distribution' and 'similarity_threshold'"
        )

    axes = subdivide_subplot(fig, ax, rows, cols)

    # print(similarity_threshold.shape, similarity_distribution.shape)

    for i in range(num_patterns):

        over_thr = (
            np.sum(similarity_distribution[i] >= similarity_threshold)
            / similarity_distribution.shape[1]
        )

        axes[i % rows, i // cols].tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        hplot = axes[i % rows, i // rows].hist(similarity_distribution[i])
        alpha_line = axes[i % rows, i // rows].vlines(
            similarity_threshold,
            ymin=0,
            ymax=np.max(hplot[0]),
            color="red",
        )
        # axes[i % rows, i // rows].text(
        #     similarity_threshold,
        #     np.max(hplot[0]) - 0.5,
        #     f"{similarity_threshold:.0f}",
        #     horizontalalignment="center",
        #     verticalalignment="center",
        #     fontsize="xx-small",
        # )
        # axes[i % rows, i // rows].text(
        #     hplot[1][0],
        #     np.max(hplot[0]) - 0.5,
        #     f"{ hplot[1][0]:.0f}",
        #     horizontalalignment="center",
        #     verticalalignment="center",
        #     fontsize="xx-small",
        # )
        # axes[i % rows, i // rows].text(
        #     hplot[1][-1],
        #     np.max(hplot[0]) - 0.5,
        #     f"{hplot[1][-1]:.0f}",
        #     horizontalalignment="center",
        #     verticalalignment="center",
        #     fontsize="xx-small",
        # )

        # print(hplot[0].shape, hplot[1].shape)
        # for j, v in enumerate(hplot[0]):
        #     bin_val = (hplot[1][j] + hplot[1][j + 1]) / 2
        #     axes[i % rows, i // rows].text(
        #         bin_val,
        #         v + 1,
        #         f"({bin_val:.2f},{v:.2f})",
        #         horizontalalignment="center",
        #         verticalalignment="center",
        #     )
        axes[i % rows, i // rows].legend(
            [hplot[2], alpha_line],
            [
                f"snaps [{hplot[1][0]:.0f},{hplot[1][-1]:.0f}]",
                f"{over_thr*100:.2f} % >= {similarity_threshold} (thr)",
            ],
            fontsize="xx-small",
        )

    ax.set_title(f"similarity distribution {pop_name} {title}")


def plot_similarity_distributions_individual(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    pop_name: str,
    title: str,
    rows: int,
    cols: int,
    similarity_threshold: np.ndarray,
    similarity_distribution: np.ndarray,
):
    """
    plot similarity distributions with individual thresholds

    :param fig: figure instance
    :param ax: axis instance
    :param rows: number of rows into which the axes/subplot is subdivided
    :param cols: number of columns into which the axes/subplot is subdivided
    :param similarity_threshold: similarity threshold (N x 1) for each pattern corresponding to the tightest lower bound on the significance level
    :param similarity_distribution: similarities between patterns and snapshots (N x C) where N is the number of patterns and C the number of cycles
    """
    num_patterns = similarity_distribution.shape[0]
    if not rows * cols >= num_patterns:
        raise ValueError(
            "product of parameters 'rows' and 'cols' must be >= N, number of patterns in 'similarity_distribution' and 'similarity_threshold'"
        )

    axes = subdivide_subplot(fig, ax, rows, cols)

    # print(similarity_threshold.shape, similarity_distribution.shape)

    for i in range(num_patterns):
        axes[i % rows, i // cols].tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        hplot = axes[i % rows, i // rows].hist(similarity_distribution[i])
        axes[i % rows, i // rows].vlines(
            similarity_threshold[i],
            ymin=0,
            ymax=np.max(hplot[0]),
            # label=f"sim thr {similarity_threshold[i]} (p{i})",
            color="red",
        )
        axes[i % rows, i // rows].text(
            similarity_threshold[i],
            np.max(hplot[0]) - 0.5,
            f"{similarity_threshold[i]:.0f}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize="xx-small",
        )
        axes[i % rows, i // rows].text(
            hplot[1][0],
            np.max(hplot[0]) - 0.5,
            f"{ hplot[1][0]:.0f}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize="xx-small",
        )
        axes[i % rows, i // rows].text(
            hplot[1][-1],
            np.max(hplot[0]) - 0.5,
            f"{hplot[1][-1]:.0f}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize="xx-small",
        )


    ax.set_title(f"similarity distribution {pop_name} {title}")


def plot_similarity_per_snapshot_over_time(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    pop_name: str,
    title: str,
    pvalues: np.ndarray,
    troughs: np.ndarray,
    stimulus_block_interval: Union[np.ndarray, Dict],
    t_start: float,
    t_end: float,
    dt: float,
    significance: float = 0.05,
):
    """
    plot similarity for the respective reference pattern over time

    :param fig: figure instance
    :param ax: axis instance
    :param pvalues: pvalues of similarity of the respective and snapshot (C,), where C is the number of snapshots
    :param troughs: troughs of the population rate (basis for the calculation of snapshots) indices in ['t_start','t_end'] with step 'dt'
    :param stimulus_block_interval: set of intervals during which stimulus is present composed of start and end time in [ms] (B x 2),
                    where B is the number of intervals and stimulus_block_interval[b,0], stimulus_block_interval[b,1] is start/end time of interval b,
                    or set of parameters incl  a subset of 'offset', 'amplitude', 'angularfrequency' and 'timeshift' characterizing an inhomogeneous poisson process
    :param t_start: start of analysis
    :param t_end: end of analysis
    :param dt: time step of simulation
    :param significance: significance level for plotting pvalues
    """
    time = np.arange(t_start, t_end + dt, dt)

    # print(similarity_threshold.shape, similarity_distribution.shape)

    pvals = pvalues
    snap_idx = np.arange(0, pvals.size)

    # snap_idx ~ index i of snapshot ~ which we will represent as the midpoint between the ith and (i+1)th troughs
    # time_troughs = (time[troughs[snap_idx + 1]] - time[troughs[snap_idx]]) / 2 + time[troughs[snap_idx]]
    time_troughs = (time[troughs[snap_idx]], time[troughs[snap_idx + 1]])

    idx = np.logical_and(time_troughs[0] >= time[0], time_troughs[1] <= time[-1])
    # print(idx)
    time_troughs = (time_troughs[0][idx], time_troughs[1][idx])
    pvals = pvals[idx]

    stimulus_lines = []

    if time_troughs[0].size > 0:
        for i in range(time_troughs[0].size):
            start_snap = time_troughs[0][i]
            end_snap = time_troughs[1][i]
            pvl = pvals[i]
            # print(start_snap, end_snap, pvl)
            ax.hlines(
                pvl,
                start_snap,
                end_snap,
                color="blue",
                label="pvals per snap",
            )

            pval_line = ax.scatter(
                [(end_snap - start_snap) / 2 + start_snap],
                [pvl],
                color="blue",
                label="pvals per snap",
                marker=".",
            )

        sign_line = ax.hlines(0.05, time[0], time[-1], color="red", label="0.05 sign.")

        zero_line = ax.hlines(0.0, time[0], time[-1], color="black", label="0.0 line")

        # instead of hlines do grids
        if isinstance(stimulus_block_interval, np.ndarray):
            for strt, end in stimulus_block_interval:
                if end >= t_start and strt <= t_end:
                    if end > t_end:
                        end = t_end
                    if strt < t_start:
                        strt = t_start
                    stim_line = ax.hlines(
                        -0.05, strt, end, color="green", label="stimulus"
                    )
                    ax.axvspan(strt, end, alpha=0.3, color="#dfdddb")
                stimulus_lines.append(stim_line)
        elif isinstance(stimulus_block_interval, dict) and any(
            [
                k in ["offset", "amplitude", "angularfrequency", "timeshift"]
                for k in stimulus_block_interval.keys()
            ]
        ):

            def inh_pois(
                t: np.ndarray,
                offset: float = 1.0,
                amplitude: float = 1.0,
                angularfrequency=2 * np.pi,
                timeshift: float = 0.0,
            ):
                """
                :param t: time in [ms]
                """
                return (
                    offset
                    + np.cos((t / 1000.0 - timeshift) * angularfrequency) * amplitude
                )

            sbi = stimulus_block_interval
            # rescale and relocate to make graph work
            stimulus_block_interval["offset"] = -0.05
            stimulus_block_interval["amplitude"] = 0.01

            rate = lambda t: inh_pois(t, **stimulus_block_interval)
            x = np.linspace(time[0], time[-1], int(1e6) + 1)

            # print(x, rate(x), stimulus_block_interval)

            line_inh_pois = ax.plot(x, rate(x), color="green", label="stimulus")
            ax.hlines(stimulus_block_interval["offset"], x[0], x[-1], color="gray")
            stimulus_lines = line_inh_pois

        else:
            raise ValueError(
                "No such stimulus_block_interval supported. Must either be a np.ndarray or a dict with subset of keys 'offset', 'amplitude', 'angularfrequency', 'timeshift'."
            )

        ax.legend(
            [pval_line, stimulus_lines[0], sign_line, zero_line],
            ["pvals_per_snap", "stimulus", f"0.05% sign", "0.0 line"],
            fontsize="xx-small",
        )

    ax.set_title(
        f"similarity per pattern {pop_name} @ 0.05 sign. (snaps {pvalues.shape[0]}) {title}",
        wrap=True,
    )


def plot_similarity_blocked_stimulus(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    pop_name: str,
    title: str,
    pvalues: np.ndarray,
    troughs: np.ndarray,
    stimulus_onset: float,
    stimulus_length: float,
    sign_snaps_pre: float,
    sign_snaps_stim: float,
    sign_snaps_post: float,
    sign_snaps_pre_far: float = None,
    sign_snaps_pre_close: float = None,
    sign_snaps_post_close: float = None,
    sign_snaps_post_far: float = None,
    significance: float = 0.05,
):
    """
    plot similarity for the respective reference pattern over time

    :param fig: figure instance
    :param ax: axis instance
    :param pvalues: pvalues of similarity of the respective and snapshot (C,), where C is the number of snapshots
    :param troughs: troughs of the population rate (basis for the calculation of snapshots) [ms]
    :param stimulus_onset: point of time of onset of stimulus [ms]
    :param stimulus_length: length of stimulus [ms]
    :param sign_snaps_pre: fraction of significant snaps in window of size parameter 'stimulus_length' before stimulus presentation
    :param sign_snaps_stim: fraction of significant snaps in window of size parameter 'stimulus_length' at stimulus presentation
    :param sign_snaps_post: fraction of significant snaps in window of size parameter 'stimulus_length' after stimulus presentation
    :param significance: significance level for plotting pvalues
    """

    detailed = (
        sign_snaps_pre_far != None
        and sign_snaps_pre_close != None
        and sign_snaps_post_close != None
        and sign_snaps_post_far != None
    )

    if (
        sign_snaps_pre_far != None
        or sign_snaps_pre_close != None
        or sign_snaps_post_close != None
        or sign_snaps_post_far != None
    ) and not detailed:
        raise ValueError(
            "Either provide all or none of sign_snaps_{pre,post}_{close,far}. passing subset not supported."
        )

    # plot devided into 5 segements: 2*sl (free)| sl (pre)| sl (stim)| sl (post)| 2*sl (free), where sl is stimulus_length
    # - assumes at least 3*sl between stimulus presentations
    # this is t_start and t_end for the exemplary 'excerpt' tb plotted
    # - note % sign snaps that are precomputed and passed are independent of this
    t_start = stimulus_onset - 3 * stimulus_length
    t_end = stimulus_onset + 4 * stimulus_length

    # print(similarity_threshold.shape, similarity_distribution.shape)

    snap_idx = np.arange(0, pvalues.size)

    # snap_idx ~ index i of snapshot ~ which we will represent as the midpoint between the ith and (i+1)th troughs
    # time_troughs = (time[troughs[snap_idx + 1]] - time[troughs[snap_idx]]) / 2 + time[troughs[snap_idx]]
    snap_beg = troughs[snap_idx]
    snap_end = troughs[snap_idx + 1]

    stimulus_lines = []

    if snap_beg.size > 0:
        idx = np.logical_and(snap_beg >= t_start, snap_end <= t_end)

        snap_beg = snap_beg[idx]
        snap_end = snap_end[idx]
        pvals = pvalues[idx]

        for pvl, sbeg, send in zip(pvals, snap_beg, snap_end):
            ax.hlines(
                pvl,
                sbeg,
                send,
                color="blue",
                label="pvals per snap",
            )

            pval_line = ax.scatter(
                [(send - sbeg) / 2 + sbeg],
                [pvl],
                color="blue",
                label="pvals per snap",
                marker=".",
            )

        sign_line = ax.hlines(0.05, t_start, t_end, color="red", label="0.05 sign.")

        zero_line = ax.hlines(0.0, t_start, t_end, color="black", label="0.0 line")

        # instead of hlines do grids
        # check whether interval in [t_start, t_end]
        stim_line = ax.hlines(
            -0.05,
            stimulus_onset,
            stimulus_onset + stimulus_length,
            color="green",
            label="stimulus",
        )
        ax.axvspan(
            stimulus_onset - stimulus_length, stimulus_onset, alpha=0.1, color="#fda335"
        )
        ax.axvspan(
            stimulus_onset, stimulus_onset + stimulus_length, alpha=0.1, color="#7f7f7f"
        )
        ax.axvspan(
            stimulus_onset + stimulus_length,
            stimulus_onset + 2 * stimulus_length,
            alpha=0.1,
            color="#3564fd",
        )
        ax.text(
            stimulus_onset - stimulus_length / 2,
            -0.06,
            "pre",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.text(
            stimulus_onset - stimulus_length / 2,
            -0.025 if not detailed else -0.015,
            f"{sign_snaps_pre:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            color="#fda335",
        )
        ax.text(
            stimulus_onset + stimulus_length / 2,
            -0.06,
            "stim",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.text(
            stimulus_onset + stimulus_length / 2,
            -0.025 if not detailed else -0.015,
            f"{sign_snaps_stim:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            color="#7f7f7f",
        )
        ax.text(
            stimulus_onset + 1.5 * stimulus_length,
            -0.06,
            "post",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.text(
            stimulus_onset + 1.5 * stimulus_length,
            -0.025 if not detailed else -0.015,
            f"{sign_snaps_post:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            color="#3564fd",
        )

        ax.text(
            stimulus_onset - 2 * stimulus_length,
            -0.025 if not detailed else -0.015,
            "% snapshots sign. at 0.05",
            horizontalalignment="center",
            verticalalignment="center",
            color="#a3a3a3",
        )

        if detailed:
            ax.vlines(
                stimulus_onset + 1.5 * stimulus_length, -0.03, -0.02, color="#3564fd"
            )

            ax.text(
                stimulus_onset + 1.25 * stimulus_length,
                -0.025,
                f"{sign_snaps_post_close:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                color="#3564fd",
            )
            ax.text(
                stimulus_onset + 1.75 * stimulus_length,
                -0.025,
                f"{sign_snaps_post_far:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                color="#3564fd",
            )
            ax.vlines(
                stimulus_onset - stimulus_length / 2, -0.03, -0.02, color="#fda335"
            )
            ax.text(
                stimulus_onset - 0.25 * stimulus_length,
                -0.025,
                f"{sign_snaps_pre_close:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                color="#fda335",
            )
            ax.text(
                stimulus_onset - 0.75 * stimulus_length,
                -0.025,
                f"{sign_snaps_pre_far:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                color="#fda335",
            )

        # ax.hlines(-0.035, t_start, t_end, color="#a3a3a3")
        # ax.hlines(-0.015, t_start, t_end, color="#a3a3a3")

        stimulus_lines.append(stim_line)

        ax.set_ylim(top=0.125)
        ax.set_yticks([0.0, 0.05])
        ax.set_xticks(
            [
                stimulus_onset - stimulus_length,
                stimulus_onset,
                stimulus_onset + stimulus_length,
                stimulus_onset + 2 * stimulus_length,
            ]
        )
        ax.set_xticklabels(
            [
                f"{- stimulus_length:.2f}",
                f"{0.0:.2f}",
                f"{stimulus_length:.2f}",
                f"{2 * stimulus_length:.2f}",
            ]
        )

        # ax.legend(
        #     [pval_line, stimulus_lines[0], sign_line, zero_line],
        #     ["pvals_per_snap", "stimulus", f"0.05% sign", "0.0 line"],
        #     fontsize="xx-small",
        # )
        ax.set_ylabel("pvalue")
        ax.set_xlabel("time relative to stimulus onset [ms]")

    ax.set_title(
        f"similarity per pattern {pop_name} @ 0.05 sign. (snaps {pvalues.shape[0]}) {title}",
        wrap=True,
    )


def plot_similarity_blocked_stimulus_sliding_window(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    pop_name: str,
    title: str,
    pvalues: np.ndarray,
    troughs: np.ndarray,
    stimulus_onset: float,
    inter_onset_interval: float,
    stimulus_length: float,
    window_length: float,
    window_step: float,
    sign_snaps: np.ndarray,  # List[float],
    mean_pvals: np.ndarray,  # List[float],
    significance: float = 0.05,
):
    """
    plot similarity for the respective reference pattern over time

    :param fig: figure instance
    :param ax: axis instance
    :param pvalues: pvalues of similarity of the respective and snapshot (C,), where C is the number of snapshots
    :param troughs: troughs of the population rate (basis for the calculation of snapshots) [ms]
    :param stimulus_onset: point of time of onset of stimulus [ms]
    :param inter_onset_interval: interval time between the onsets of any two subsequent stimulus presentations [ms]
    :param stimulus_length: length of stimulus [ms]
    :param window_length: length of sliding window [ms]
    :param window_step: step size of sliding window [ms]
    :param sign_snaps: fraction of significant snapshots at significance level over windows
                        (window order: [so, so + wl], ..., [so + ioi - wl, so + ioi],
                         where so is stimulus_onset, wl is window_length, ioi is inter_onset_interval)
    :param mean_pvals: mean of the pvalues over windows (same order as 'sign_snaps')
    :param significance: significance level for plotting pvalues
    """

    axes = subdivide_subplot(fig, ax, 2, 1).squeeze()

    # this is t_start and t_end for the exemplary 'excerpt' tb plotted
    # - note % sign snaps that are precomputed and passed are independent of this
    num_windows = round(inter_onset_interval / window_step)
    wl, wr = int(np.floor(num_windows / 2).item()), int(np.ceil(num_windows / 2).item())
    # if # windows even shift window starting at onset to the left by one (effects: -|-|stim|+ -> -|stim|+|+, where stim starts with onset)
    if wl == wr and wl > 0:
        wl -= 1
        wr += 1
    t_start = stimulus_onset - wl * window_step
    t_end = stimulus_onset + wr * window_step

    # reorder sign_snaps and pvalues
    sign_snaps = np.hstack((sign_snaps[-wl:], sign_snaps[0:wr]))
    mean_pvals = np.hstack((mean_pvals[-wl:], mean_pvals[0:wr]))

    # print(similarity_threshold.shape, similarity_distribution.shape)
    snap_idx = np.arange(0, pvalues.size)

    # snap_idx ~ index i of snapshot ~ which we will represent as the midpoint between the ith and (i+1)th troughs
    # time_troughs = (time[troughs[snap_idx + 1]] - time[troughs[snap_idx]]) / 2 + time[troughs[snap_idx]]
    snap_beg = troughs[snap_idx]
    snap_end = troughs[snap_idx + 1]

    if snap_beg.size > 0:
        idx = np.logical_and(snap_beg >= t_start, snap_end <= t_end)

        snap_beg = snap_beg[idx]
        snap_end = snap_end[idx]
        pvals = pvalues[idx]

        axes[0].hlines(
            0.05,
            t_start - stimulus_onset,
            t_end - stimulus_onset,
            color="red",
            label="0.05 sign.",
        )

        axes[0].hlines(
            0.0,
            t_start - stimulus_onset,
            t_end - stimulus_onset,
            color="black",
            label="0.0 line",
        )

        for pvl, sbeg, send in zip(pvals, snap_beg, snap_end):
            # resolve the exemplary excerpt time values to the stimulus onset relative time
            sbeg -= stimulus_onset
            send -= stimulus_onset
            axes[0].hlines(
                pvl,
                sbeg,
                send,
                color="blue",
                label="pvals per snap",
            )

            axes[0].scatter(
                [(send - sbeg) / 2 + sbeg],
                [pvl],
                color="blue",
                label="pvals per snap",
                marker=".",
            )

        # stimulus
        # axes[0].hlines(
        #     -0.05,
        #     0.0,
        #     stimulus_length,
        #     color="green",
        #     label="stimulus",
        # )

        axes[1].hlines(
            0.0,
            t_start - stimulus_onset,
            t_end - stimulus_onset,
            color="black",
            label="0.0 line",
        )

        time = np.arange(t_start, t_end, window_step) - stimulus_onset

        # axes[1].plot(time, sign_snaps, label=f"% sign. snapshots")
        axes[1].scatter(
            time,
            sign_snaps,
            label=f"% sign. snapshots",
            marker=".",
            s=(72.0 / fig.dpi) ** 2,
        )

        # axes[1].plot(time, mean_pvals, label="mean pvalues")
        axes[1].scatter(
            time, mean_pvals, label="mean pvalues", marker=".", s=(72.0 / fig.dpi) ** 2
        )

        # grids
        axes[0].axvspan(0.0, stimulus_length, alpha=0.1, color="#7f7f7f")
        axes[1].axvspan(0.0, stimulus_length, alpha=0.1, color="#7f7f7f")
        axes[1].text(
            stimulus_length / 2,
            1.1,
            f"stimulus",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize="small",
            color="#7f7f7f",
        )
        axes[1].set_ylim(top=1.15)

        axes[0].set_yticks([0.0, 0.05, round(np.max(pvalues) + 0.02, 2)])

        axes[0].set_ylabel("pvalue")
        axes[1].legend(fontsize="xx-small")
        axes[1].set_xlabel("time relative to stimulus onset [ms]")

    ax.set_title(
        f"similarity per pattern {pop_name} @ 0.05 sign. (snaps {pvalues.shape[0]}) {title}",
        wrap=True,
    )


def plot_similarity_blocked_stimulus_sliding_window_delta(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    stimulus_onset: float,
    inter_onset_interval: float,
    stimulus_length: float,
    window_step: float,
    sign_snaps: np.ndarray = None,
    mean_pvals: np.ndarray = None,
    mean_similarity: np.ndarray = None,
    sign_snaps_unweighted: np.ndarray = None,
    mean_pvals_unweighted: np.ndarray = None,
    mean_similarity_unweighted: np.ndarray = None,
    zoomed: bool = False,
    excerpt_size: float = 100.0,
):
    """
    plot significant any combination of snapshots, mean pvalues and mean similarity for the respective reference pattern using a sliding window
    for the same experiment configuration simulated with and without (random connectivity) weights and their delta

    :param fig: figure instance
    :param ax: axis instance
    :param stimulus_onset: point of time of onset of stimulus [ms]
    :param inter_onset_interval: interval time between the onsets of any two subsequent stimulus presentations [ms]
    :param stimulus_length: length of stimulus [ms]
    :param window_step: step size of sliding window [ms]
    :param sign_snaps: fraction of significant snapshots at significance level over window
                        (window order: [so, so + wl], ..., [so + ioi - wl, so + ioi],
                         where so is stimulus_onset, wl is window_length, ioi is inter_onset_interval)
    :param mean_pvals: mean of the pvalues over window (same order as 'sign_snaps')
    :param mean_similarity: mean of the similarity over window (same order as 'sign_snaps')
    :param sign_snaps_unweighted: fraction of significant snapshots at significance level over windows for unweighted simulation (control)
                        (window order: [so, so + wl], ..., [so + ioi - wl, so + ioi],
                         where so is stimulus_onset, wl is window_length, ioi is inter_onset_interval) for unweighted simulation (control)
    :param mean_pvals_unweighted: mean of the pvalues over windows (same order as 'sign_snaps')
    :param mean_similarity_unweighted: mean of the similarity over window (same order as 'sign_snaps')
    :param zoomed: whether or not to restrict plot to excerpts around stimulus onset and end
    :param excerpt_size: size of the excerpt [ms] around stimulus onset and stimulus end - considered only if zoomed set to True
    """

    # color palette
    color_ss = sequential_palette("Reds", 3)
    color_mp = sequential_palette("Blues", 3)
    color_ms = sequential_palette("Greys", 3)
    c_ss = next(color_ss)
    c_ssu = next(color_ss)
    c_mp = next(color_mp)
    c_mpu = next(color_mp)
    c_ms = next(color_ms)
    c_msu = next(color_ms)

    args = ["sign_snaps", "mean_pvals", "mean_similarity"]

    # ensure that at least one of the three: sign snaps, mean pval, mean sim is provided and that unweighted args match this

    context = locals()
    if not any([not isinstance(context[e], type(None)) for e in args]):
        raise ValueError(f"provide at least one of the three args {args}")
    if not all(
        [
            not isinstance(context[f"{e}_unweighted"], type(None))
            for e in args
            if not isinstance(context[e], type(None))
        ]
    ):
        raise ValueError(
            "not all args in ['sign_snaps', 'mean_pvals', 'mean_similarity'] are also provided for unweighted simulation ('_unweighted')"
        )

    axes = subdivide_subplot(fig, ax, 2, 1).squeeze()

    # this is t_start and t_end for the exemplary 'excerpt' tb plotted
    # - note % sign snaps that are precomputed and passed are independent of this
    num_windows = round(inter_onset_interval / window_step)
    if num_windows > sign_snaps.size:
        raise ValueError(
            f"num_windows := 'inter_onset_interval' / 'window_step' must be <= 'sign_snaps'.size ({inter_onset_interval}/{window_step} "
            + f"= {inter_onset_interval/window_step} <=? {sign_snaps.size}).Adapt 'window_step' such that inequality holds."
        )

    relax_windows = round((inter_onset_interval - stimulus_length) / window_step)
    wl, wr = int(np.floor(relax_windows / 2).item()), int(
        np.ceil(relax_windows / 2).item()
    )
    # add stimulus windows
    wr += num_windows - relax_windows

    t_start = stimulus_onset - wl * window_step
    t_end = stimulus_onset + wr * window_step

    # reorder sign_snaps and pvalues
    # for e in zip([sign_snaps, mean_pvals, mean_similarity, sign_snaps_unweighted, mean_pvals_unweighted, mean_similarity_unweighted]):
    if not isinstance(sign_snaps, type(None)):
        sign_snaps = np.hstack((sign_snaps[-wl:], sign_snaps[0:wr]))
    if not isinstance(mean_pvals, type(None)):
        mean_pvals = np.hstack((mean_pvals[-wl:], mean_pvals[0:wr]))
    if not isinstance(mean_similarity, type(None)):
        mean_similarity = np.hstack((mean_similarity[-wl:], mean_similarity[0:wr]))
    if not isinstance(sign_snaps_unweighted, type(None)):
        sign_snaps_unweighted = np.hstack(
            (sign_snaps_unweighted[-wl:], sign_snaps_unweighted[0:wr])
        )
    if not isinstance(mean_pvals_unweighted, type(None)):
        mean_pvals_unweighted = np.hstack(
            (mean_pvals_unweighted[-wl:], mean_pvals_unweighted[0:wr])
        )
    if not isinstance(mean_similarity_unweighted, type(None)):
        mean_similarity_unweighted = np.hstack(
            (mean_similarity_unweighted[-wl:], mean_similarity_unweighted[0:wr])
        )


    time = np.arange(t_start, t_end, window_step) - stimulus_onset

    plotst = []
    plotsb = []

    if not zoomed:
        plx = axes[0].hlines(
            0.0,
            t_start - stimulus_onset,
            t_end - stimulus_onset,
            color="black",
        )
        plotst.append((plx, "0 line"))
        # print(time.shape, sign_snaps.shape, num_windows)

        if not isinstance(sign_snaps, type(None)):
            plx = axes[0].scatter(
                time,
                sign_snaps,
                color=c_ss,
                alpha=0.3,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
            )
            plotst.append((plx, f"% sign. snapshots"))
            plx = axes[0].scatter(
                time,
                sign_snaps_unweighted,
                color=c_ssu,
                alpha=0.3,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
            )
            plotst.append((plx, f"% sign. snapshots (unweighted)"))

        if not isinstance(mean_pvals, type(None)):
            plx = axes[0].scatter(
                time,
                mean_pvals,
                color=c_mp,
                alpha=0.3,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
            )
            plotst.append((plx, "mean pvalues"))

            plx = axes[0].scatter(
                time,
                mean_pvals_unweighted,
                color=c_mpu,
                alpha=0.3,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
            )
            plotst.append((plx, "mean pvalues (unweighted)"))

        if not isinstance(mean_similarity, type(None)):
            twin = axes[0].twinx()
            plx = twin.scatter(
                time,
                mean_similarity,
                color=c_ms,
                alpha=0.3,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
            )
            plotst.append((plx, "mean similarity"))

            plx = twin.scatter(
                time,
                mean_similarity_unweighted,
                color=c_msu,
                alpha=0.3,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
            )
            plotst.append((plx, "mean simlarity (unweighted)"))
            twin.set_ylabel("similarity", fontsize="x-small")
            twin.tick_params(axis="both", which="both", labelsize="xx-small")

        # grids
        axes[0].axvspan(0.0, stimulus_length, alpha=0.1, color="#7f7f7f")
        axes[1].axvspan(0.0, stimulus_length, alpha=0.1, color="#7f7f7f")

        axes[1].text(
            stimulus_length / 2,
            -0.1,
            f"stimulus",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize="small",
            color="#7f7f7f",
        )

        if not isinstance(sign_snaps, type(None)):
            plx = axes[1].scatter(
                time,
                sign_snaps - sign_snaps_unweighted,
                color=c_ss,
                alpha=0.3,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
            )
            plotsb.append((plx, f"delta % sign. snapshots"))

        if not isinstance(mean_pvals, type(None)):
            plx = axes[1].scatter(
                time,
                mean_pvals - mean_pvals_unweighted,
                color=c_mp,
                alpha=0.3,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
            )
            plotsb.append((plx, "delta mean pvalues"))

        if not isinstance(mean_similarity, type(None)):
            twin_delta = axes[1].twinx()
            plx = twin_delta.scatter(
                time,
                mean_similarity - mean_similarity_unweighted,
                color=c_ms,
                alpha=0.3,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
            )
            plotsb.append((plx, "delta mean similarity"))

            twin_delta.set_ylabel("delta similarity", fontsize="x-small")
            twin_delta.tick_params(axis="both", which="both", labelsize="xx-small")

        # axes[1].set_ylim(top=1.15)
        axes[0].set_ylabel("pvalue/fraction", fontsize="x-small")
        axes[1].set_ylabel(f"delta", fontsize="x-small")
        axes[1].set_xlabel("time relative to stimulus onset [ms]", fontsize="x-small")

        axes[0].tick_params(axis="both", which="both", labelsize="xx-small")
        axes[1].tick_params(axis="both", which="both", labelsize="xx-small")

        # raise ValueError(plotst, list(zip(*plotst)))

        legt = axes[0].legend(*list(zip(*plotst)), fontsize="xx-small")
        legb = axes[1].legend(*list(zip(*plotsb)), fontsize="xx-small")

        for leg in [legt, legb]:
            for lhdl in leg.legendHandles:
                lhdl.set_alpha(1)

    # shorten to excerpts
    else:

        num_win_excerpt = round(excerpt_size / window_step)
        num_win_stimulus = round(stimulus_length / window_step)


        # select [-num_win_excerpt,0] [num_win_stimulus, num_win_stimulus + num_win_excerpt]
        def create_excerpt(x, num_excerpt, num_stim):
            # 3 segments [-exerpt,0], [0, stim_end], [stim_end, stim_end+excerpt]
            # x = np.hstack((x[wl : wl + wr], x[0:wl]))
            # print(f"wl {wl}, num_excerpt {num_excerpt}, num stim {num_stim}, wr {wr}")
            return (
                x[
                    max(wl - num_excerpt // 2, 0) : wl
                    + min(num_excerpt // 2, num_stim // 2)
                ],
                x[
                    wl
                    + num_stim
                    - min(num_excerpt // 2, num_stim // 2) : wl
                    + num_stim
                    + num_excerpt // 2
                ],
            )

        data = {}
        if not isinstance(sign_snaps, type(None)):
            data["sl"], data["sr"] = create_excerpt(
                sign_snaps, num_win_excerpt, num_win_stimulus
            )
            data["slu"], data["sru"] = create_excerpt(
                sign_snaps_unweighted, num_win_excerpt, num_win_stimulus
            )
        if not isinstance(mean_pvals, type(None)):
            data["mpl"], data["mpr"] = create_excerpt(
                mean_pvals, num_win_excerpt, num_win_stimulus
            )
            data["mplu"], data["mpru"] = create_excerpt(
                mean_pvals_unweighted, num_win_excerpt, num_win_stimulus
            )
        if not isinstance(mean_similarity, type(None)):
            data["msl"], data["msr"] = create_excerpt(
                mean_similarity, num_win_excerpt, num_win_stimulus
            )
            data["mslu"], data["msru"] = create_excerpt(
                mean_similarity_unweighted, num_win_excerpt, num_win_stimulus
            )
        time_left, time_right = create_excerpt(time, num_win_excerpt, num_win_stimulus)

        # print(time.shape, time_left.shape, time_right.shape)

        axes_top = subdivide_subplot(fig, axes[0], 1, 2).squeeze()
        axes_bottom = subdivide_subplot(fig, axes[1], 1, 2).squeeze()

        # axes[0].hlines(
        #     0.0,
        #     t_start - stimulus_onset,
        #     t_end - stimulus_onset,
        #     color="black",
        #     label="0.0 line",
        # )


        plotst = [[], []]
        plotsb = [[], []]

        twin_top_axs = []
        twin_bottom_axs = []

        for i, (tm, lr) in enumerate(zip([time_left, time_right], ["l", "r"])):

            plx = axes_top[i].hlines(
                0.0,
                tm[0],
                tm[-1],
                color="black",
            )
            if lr == "l":
                plotst[i].append((plx, "0 line"))
            else:
                plotst[i].append((plx, "0 line"))

            if not isinstance(sign_snaps, type(None)):
                ss = data[f"s{lr}"]
                ssu = data[f"s{lr}u"]

                plx = axes_top[i].scatter(
                    tm,
                    ss,
                    color=c_ss,
                    marker=".",
                    alpha=0.3,
                    s=(72.0 / fig.dpi) ** 2,
                )
                if lr == "l":
                    plotst[i].append((plx, f"% sign. snapshots"))
                else:
                    plotst[i].append((plx, f"% sign. snapshots"))

                plx = axes_top[i].scatter(
                    tm,
                    ssu,
                    color=c_ssu,
                    marker=".",
                    s=(72.0 / fig.dpi) ** 2,
                )
                if lr == "l":
                    plotst[i].append((plx, f"% sign. snapshots (unweighted)"))
                else:
                    plotst[i].append((plx, f"% sign. snapshots (unweighted)"))

                plx = axes_bottom[i].scatter(
                    tm,
                    ss - ssu,
                    color=c_ss,
                    alpha=0.3,
                    marker=".",
                    s=(72.0 / fig.dpi) ** 2,
                )
                if lr == "l":
                    plotsb[i].append((plx, f"delta % sign. snapshots"))
                else:
                    plotsb[i].append((plx, f"delta % sign. snapshots"))

            if not isinstance(mean_pvals, type(None)):
                mp = data[f"mp{lr}"]
                mpu = data[f"mp{lr}u"]
                plx = axes_top[i].scatter(
                    tm,
                    mp,
                    color=c_mp,
                    alpha=0.3,
                    label="mean pvalues",
                    marker=".",
                    s=(72.0 / fig.dpi) ** 2,
                )
                if lr == "l":
                    plotst[i].append((plx, f"mean pvalues"))
                else:
                    plotst[i].append((plx, f"mean pvalues"))

                plx = axes_top[i].scatter(
                    tm,
                    mpu,
                    color=c_mpu,
                    alpha=0.3,
                    label="mean pvalues (unweighted)",
                    marker=".",
                    s=(72.0 / fig.dpi) ** 2,
                )
                if lr == "l":
                    plotst[i].append((plx, f"mean pvalues (unweighted)"))
                else:
                    plotst[i].append((plx, f"mean pvalues (unweighted)"))

                plx = axes_bottom[i].scatter(
                    tm,
                    mp - mpu,
                    color=c_mp,
                    alpha=0.3,
                    label="delta mean pvalues",
                    marker=".",
                    s=(72.0 / fig.dpi) ** 2,
                )

                if lr == "l":
                    plotsb[i].append((plx, f"delta mean pvalues"))
                else:
                    plotsb[i].append((plx, f"delta mean pvalues"))

            if not isinstance(mean_similarity, type(None)):
                ms = data[f"ms{lr}"]
                msu = data[f"ms{lr}u"]

                twin_top = axes_top[i].twinx()
                twin_bottom = axes_bottom[i].twinx()
                twin_top_axs.append(twin_top)
                twin_bottom_axs.append(twin_bottom)

                if i == 0:
                    twin_top.tick_params(
                        axis="y",
                        which="both",
                        bottom=False,
                        top=False,
                        left=False,
                        right=False,
                        labelbottom=False,
                        labeltop=False,
                        labelleft=False,
                        labelright=False,
                    )
                    twin_bottom.tick_params(
                        axis="y",
                        which="both",
                        bottom=False,
                        top=False,
                        left=False,
                        right=False,
                        labelbottom=False,
                        labeltop=False,
                        labelleft=False,
                        labelright=False,
                    )
                else:
                    twin_top.set_ylabel("similarity", fontsize="x-small")
                    twin_bottom.set_ylabel("delta similarity", fontsize="x-small")
                    twin_top.tick_params(
                        axis="both", which="both", labelsize="xx-small"
                    )
                    twin_bottom.tick_params(
                        axis="both", which="both", labelsize="xx-small"
                    )
                plx = twin_top.scatter(
                    tm,
                    ms,
                    color=c_ms,
                    alpha=0.3,
                    marker=".",
                    s=(72.0 / fig.dpi) ** 2,
                )
                if lr == "l":
                    plotst[i].append((plx, f"mean similarity"))
                else:
                    plotst[i].append((plx, f"mean similarity"))

                plx = twin_top.scatter(
                    tm,
                    msu,
                    color=c_msu,
                    alpha=0.3,
                    marker=".",
                    s=(72.0 / fig.dpi) ** 2,
                )
                if lr == "l":
                    plotst[i].append((plx, f"mean similarity (unweighted)"))
                else:
                    plotst[i].append((plx, f"mean similarity (unweighted)"))

                # bottom
                plx = twin_bottom.scatter(
                    tm,
                    ms - msu,
                    color=c_ms,
                    alpha=0.3,
                    marker=".",
                    s=(72.0 / fig.dpi) ** 2,
                )

                if lr == "l":
                    plotsb[i].append((plx, f"delta mean similarity"))
                else:
                    plotsb[i].append((plx, f"delta mean similarity"))

            axes_bottom[i].set_xlabel(
                "time relative to stimulus onset [ms]", fontsize="x-small"
            )

            if i == 0:
                axes_top[i].set_ylabel("pvalue/fraction", fontsize="x-small")
                axes_bottom[i].set_ylabel("delta pvalue/fraction", fontsize="x-small")

            axes_top[i].tick_params(axis="both", which="both", labelsize="xx-small")
            axes_bottom[i].tick_params(axis="both", which="both", labelsize="xx-small")

            legt = axes_top[i].legend(*list(zip(*plotst[i])), fontsize="xx-small")
            legb = axes_bottom[i].legend(*list(zip(*plotsb[i])), fontsize="xx-small")
            for leg in [legt, legb]:
                for lhdl in leg.legendHandles:
                    lhdl.set_alpha(1)

        # standardize scale across left and right plot for twin y axis with shared x axis
        for twin, twin_delta in [twin_top_axs, twin_bottom_axs]:
            ymn, ymx = min(twin.get_ylim()[0], twin_delta.get_ylim()[0]), max(
                twin.get_ylim()[1], twin_delta.get_ylim()[1]
            )
            twin.set_ylim(ymn, ymx)
            twin_delta.set_ylim(ymn, ymx)

        # grids
        axes_top[0].axvspan(
            time_left[-min(num_win_excerpt // 2, num_win_stimulus // 2)],
            time_left[-1],
            alpha=0.1,
            color="#7f7f7f",
        )
        axes_bottom[0].axvspan(
            time_left[-min(num_win_excerpt // 2, num_win_stimulus // 2)],
            time_left[-1],
            alpha=0.1,
            color="#7f7f7f",
        )

        axes_top[1].axvspan(
            time_right[0],
            time_right[min(num_win_excerpt // 2, num_win_stimulus // 2)],
            alpha=0.1,
            color="#7f7f7f",
        )
        axes_bottom[1].axvspan(
            time_right[0],
            time_right[min(num_win_excerpt // 2, num_win_stimulus // 2)],
            alpha=0.1,
            color="#7f7f7f",
        )

        # axes[1].text(
        #     stimulus_length / 2,
        #     -0.1,
        #     f"stimulus",
        #     horizontalalignment="center",
        #     verticalalignment="center",
        #     fontsize="small",
        #     color="#7f7f7f",
        # )

        axes_bottom[0].set_ylabel(f"delta", fontsize="x-small")

        # deal with the y axis of the right and lefthand plot
        for axx in [axes_top[1], axes_bottom[1]]:
            axx.tick_params(
                axis="y",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
            )

            # move y axis of righthand plot the left
            # ax.yaxis.set_label_position("right")
            # ax.yaxis.tick_right()

        for axx in [axes_top, axes_bottom]:
            ymn, ymx = min(axx[0].get_ylim()[0], axx[1].get_ylim()[0]), max(
                axx[0].get_ylim()[1], axx[1].get_ylim()[1]
            )
            axx[0].set_ylim(ymn, ymx)
            axx[1].set_ylim(ymn, ymx)


def plot_similarity_per_pattern(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    pop_name: str,
    title: str,
    rows: int,
    cols: int,
    pvalues: np.ndarray,
    significance: float = 0.05,
):
    """
    plot similarity per pattern

    :param fig: figure instance
    :param ax: axis instance
    :param rows: number of rows into which the axes/subplot is subdivided
    :param cols: number of columns into which the axes/subplot is subdivided
    :param pvalues: pvalues of similarity per pattern and snapshot (N x C), where N is number of patterns and C is snapshots
    :param significance: significance level for plotting pvalues
    """
    num_patterns = pvalues.shape[0]
    if not rows * cols >= num_patterns:
        raise ValueError(
            "product of parameters 'rows' and 'cols' must be >= N, number of patterns in 'similarity_distribution' and 'similarity_threshold'"
        )

    axes = subdivide_subplot(fig, ax, rows, cols)

    # print(similarity_threshold.shape, similarity_distribution.shape)

    for i in range(num_patterns):
        pvals = pvalues[i]
        idx = np.logical_and(pvals <= significance, pvals >= 0.0)
        snaps = np.where(idx)[0]
        pvals = pvals[idx]
        axes[i % rows, i // cols].tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )

        if snaps.size > 0:
            axes[i % rows, i // rows].scatter(
                snaps,
                pvals,
                color="blue",
                label="pvals per snap",
                marker=".",
            )
            axes[i % rows, i // rows].hlines(
                0.05, 0, idx.size, color="red", label="0.05 sign."
            )
            axes[i % rows, i // rows].hlines(
                0.0, 0, idx.size, color="black", label="0.0 line"
            )
            axes[i % rows, i // rows].legend(fontsize="xx-small")

    ax.set_title(
        f"similarity per pattern {pop_name} (snaps {pvalues.shape[1]}) {title}"
    )


def plot_pairwise_similarity(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    pop_name: str,
    title: str,
    rows: int,
    cols: int,
    pvalues: np.ndarray,
    significance: float = 0.05,
):
    """
    plot pairwise similarity
    :param fig: figure instance
    :param ax: axis instance
    :param rows: number of rows into which the axes/subplot is subdivided
    :param cols: number of columns into which the axes/subplot is subdivided
    :param pvalues: pvalues of similarity per pattern and snapshot (N x C), where N is number of patterns and C is snapshots
    :param significance: significance level for plotting pvalues
    """
    num_patterns = pvalues.shape[0]
    if not rows * cols >= num_patterns:
        raise ValueError(
            "product of parameters 'rows' and 'cols' must be >= N, number of patterns in 'similarity_distribution' and 'similarity_threshold'"
        )

    axes = subdivide_subplot(fig, ax, rows, cols)

    # print(similarity_threshold.shape, similarity_distribution.shape)

    for i in range(num_patterns):
        pvals = pvalues[i]
        pairwise_pvals = np.vstack(
            (
                np.logical_and(pvals[0:-1] <= significance, pvals[0:-1] >= 0.0),
                np.logical_and(pvals[1:] <= significance, pvals[1:] >= 0.0),
            ),
        )
        pairwise_idx = np.sum(pairwise_pvals, axis=0) == 2
        snaps = np.where(pairwise_idx)[0]

        axes[i % rows, i // cols].tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )

        if snaps.size > 0:
            axes[i % rows, i // rows].scatter(
                snaps,
                snaps + 1,
                color="blue",
                label="pvals per snap",
                marker=".",
            )
            axes[i % rows, i // rows].set_axisbelow(True)
            axes[i % rows, i // rows].grid(color="gray", linestyle="dashed")
            axes[i % rows, i // rows].set_xlim(left=0, right=pvals.size)
            axes[i % rows, i // rows].set_ylim(bottom=0, top=pvals.size)

            axes[i % rows, i // rows].legend(fontsize="xx-small")

    ax.set_title(
        f"pairwise similarity {pop_name} (snaps {pvalues.shape[1]}) layout: bl->tr {title}"
    )


def plot_total_synaptic_conductance(
    fig: plt.Figure,
    ax: plt.Axes.axes,
    pop_name: str,
    title: str,
    total_synaptic_conductance: np.ndarray,
):
    """
    plot total synaptic conductance

    :param fig: figure instance
    :param ax: axis instance
    :param pop_name: name of population
    :param total_synaptic_conductance: total synaptic conductance per neuron
    """
    ax.hist(total_synaptic_conductance)
    ax.set_title(f"total syn. cond. {pop_name} {title}")


def plot_spike_train(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    pop_name: List[str],
    train: Dict[str, Tuple[np.ndarray, np.ndarray]],
    size: Dict[str, int],
    color: Dict[str, str] = {},
    title: str = "",
):
    """
    id to spike train plot for neuron populations

    :param fig: figure instance
    :param ax: axis instance
    :param pop_name: populations tb plotted in order
    :param train: spike train by neuron population as a Tuple of ids and spikes (where neuron ids[i] spiked at spike time spikes[i])
    :param size: popullation size by neuron population
    :param color: color by neuron population (opt.)
    :param title: title of plot
    """
    if not all([pn in train.keys() and pn in size.keys() for pn in pop_name]):
        raise ValueError(
            f"For some of the populations in pop_names no spike trainis provided. Is {pop_name}."
        )

    offset = 0
    for pn in pop_name:
        ids, spikes = train[pn]
        ax.scatter(
            spikes,
            ids + offset,
            color=color[pn] if pn in color.keys() else None,
            marker=".",
            s=(72.0 / fig.dpi) ** 2,
            label=pn,
        )

        offset += size[pn]

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("ids")
    ax.legend()
    if title != "":
        ax.set_title(f"spike train {pop_name} {title}")
    else:
        ax.set_title(f"spike train {pop_name}")

    # significantly faster than creating a legend for large datasets
    # data points in yrange 10-90%: 1:4 split: 80/5 = 16 -> [10, 73], [74, 90] -> mid points 42, mid points 82
    # ax.annotate(
    #    "E", (0.985, 0.42), xycoords="axes fraction", color=colors["E"], weight="bold"
    # )
    # ax.annotate(
    #    "I", (0.985, 0.82), xycoords="axes fraction", color=colors["I"], weight="bold"
    # )


def plot_variable(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    times,
    pop_name: List[str],
    ids: Dict[str, List[int]],
    variable: Dict[str, np.ndarray],
    color: Dict[str, Iterator],
    xlabel: str,
    ylabel: str,
    title: str = "",
):
    """
    plot of voltages of excitatory and inhibitory population

    :param fig: figure instance
    :param ax: axis instance
    :param times: time points
    :param pop_name: names of populations tb plotted in order
    :param ids_e: ids of neurons by population
    :param variable: variable by neuron id and population
    :param color: iterator over colors by population
    :param xlabel: label for the x axis
    :param ylabel: label for the y axis
    :param title: title of plot
    """

    for pn in pop_name:
        for i, id_ in enumerate(ids[pn]):
            v = variable[pn][i]

            ax.plot(
                times,
                v,
                label=f"{pn}: {id_}",
                color=next(color[pn]) if pn in color.keys() else None,
                alpha=0.7,
            )

    ax.set_ylabel(ylabel)  # "potential [mV]")
    ax.set_xlabel(xlabel)  # "time [ms]")
    ax.legend()
    if title != "":
        ax.set_title(f"{ylabel} {pop_name} {title}")
    else:
        ax.set_title(f"{ylabel} {pop_name}")


def plot_instantaneous_rate(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    times: np.ndarray,
    instantaneous_rate: Dict[str, Tuple[np.ndarray, str]],
    color: str,
    title: str = "",
):
    """
    plot instantaneous rate of populations

    :param fig: figure instance
    :param ax: axis instance
    :param times: time points
    :param instantaneous_rate: instantaneous population rate over time by population
    :param color: color by population
    :param title: title of plot
    """
    # dont use res here if anything use convolution or sth
    for pop_name, (pop_rate, unit) in instantaneous_rate.items():
        ax.plot(times, pop_rate, alpha=0.5, label=pop_name, color=color[pop_name])
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(f"instantaneous rate [{unit}]")
    if title != "":
        ax.set_title(f"inst pop rate {title}")
    else:
        ax.set_title(f"inst pop rate")
    ax.legend()


def plot_smoothed_rate(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    times: np.ndarray,
    smoothed_rate: Dict[str, Tuple[np.ndarray, str, str]],
    color: Dict[str, str],
    title: str = "",
    sync_freq: Dict[str, float] = {},
    snr: Dict[str, float] = {},
):
    """
    plot smooth rate of populations

    :param fig: figure instance
    :param ax: axis instance
    :param times: time points
    :param smoothed_rate: smoothed population rate over time by population
    :param color: color by population
    :param title: title of plot
    :param sync_freq: synchronization frequency of the population rate by population
    :param snr: snr of the population rate by population
    """

    for pop_name, (rate, mode, unit) in smoothed_rate.items():
        label = (
            f"{pop_name} (mode {mode}) sync freq:{sync_freq[pop_name]:.2f}[Hz]"
            if pop_name in sync_freq
            else f"{pop_name} (mode {mode})"
        )
        label = f"{label} snr:{snr[pop_name]:.2f}" if pop_name in snr else label
        ax.plot(
            times,
            rate,
            label=label,
            color=color[pop_name],
        )
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(f"smooth rate [{unit}]")
    if title != "":
        ax.set_title(f"smoothed pop rate {title}")
    else:
        ax.set_title(f"smoothed pop rate")

    ax.legend()


def plot_cell_rate(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    pop_name: str,
    cell_rate: np.ndarray,
    unit: str,
    ids: np.ndarray,
    color: Dict[str, str],
    title: str = "",
):
    """
    plot smooth rate of populations

    :param fig: figure instance
    :param ax: axis instance
    :param cell_rate: cell rate and ids per neuron by population
    :param unit: unit of cell rate
    :param ids: neuron ids corresponding to cell rates
    :param color: color by population
    :param title: title of plot
    """
    # dont use res here if anything use convolution or sth
    ax.scatter(
        ids,
        cell_rate,
        label=f"{pop_name} mean: {np.mean(cell_rate):.2f} max: {np.max(cell_rate):.2f}",
        color=color[pop_name],
        marker=".",
        s=(72.0 / fig.dpi) ** 2,
    )
    ax.set_xlabel("ids")
    ax.set_ylabel(f"cell rate [{unit}]")
    if title != "":
        ax.set_title(f"cell rate {title}")
    else:
        ax.set_title(f"cell rate")

    ax.legend()


def plot_multitaper_spectrum(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    frequency,
    psd,
    color: str = None,
    pop_name: str = None,
    title: str = "",
):
    """
    Plot multitaper power spectral density of a population rate for an entire time series
    - log power against frequency

    :param fig: figure instance
    :param ax: axis instance
    :param frequency: frequencies corresponding to spectral densities in param psd
    :param psd: power spectral density
    :param color: color of psd plot
    :param pop_name: used for specifying the title (optional)
    :param title: title of plot
    """

    ax.plot(frequency, np.log(psd), color=color)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("log power ")

    tl = "Spectrum "
    if pop_name != None:
        tl += f" (pop. {pop_name})"
    if title != "":
        tl += f" {title}"
    ax.set_title(tl)


def plot_multitaper_spectogram(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    frequency: np.ndarray,
    psd: np.ndarray,
    t_start: float,
    t_end: float,
    pop_name: str = None,
    title: str = "",
):
    """
    Plot multitaper spectogram of a population rate developing over time (separate psds computed with a sliding window)
    - power against frequency and time

    :param fig: figure instance
    :param ax: axis instance
    :param frequency: corresponding frequencies to densities in param psd[:,i] for any specific interval i
    :param psd: power spectral density (shape: (nfft/2, intervals) ~ rows -> psd, col -> time) (see also :func:`analysis.multitaper_power_spectral_density`)
    :param t_start: start time of the time series
    :param t_end: end time of the time series
    :param pop_name: used for specifying the title (optional)
    :param title: title of plot
    """

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)

    im = ax.imshow(
        psd,
        origin="lower",
        extent=[t_start, t_end, frequency[0], frequency[-1]],
        aspect="auto",
        cmap="jet",
    )

    fig.colorbar(im, cax=cax, orientation="vertical", label="power")

    # factors = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # ax.set_xticks(factors * psd.shape[0])
    # ax.set_xticklabels([str(int(e)) for e in factors * t])
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("f [Hz]")

    tl = "Spectogram"
    if pop_name != None:
        tl += f" (pop. {pop_name})"
    if title != "":
        tl += f" {title}"
    ax.set_title(tl)


def plot_synchronization_regimes():
    """
    Synchronization features across the regimes Ing, Bifurcation, Ping
    """
    pass


class ExperimentPlotter:
    """
    Plot data from :class:`BrianExperiment.BrianExperiment` and analysis data from :class:`ExperimentAnalysis.ExperimentAnalysis`

    Example for plotting

     .. code-block:: python

        from BrianExperiment import BrianExperiment
        from analysis import ExperimentAnalysis
        from plot import ExperimentPlotter
        from persistence import FileMap
        from utils import TestEnv

        with TestEnv():
            with BrianExperiment(persist=True, path="file.h5") as exp:
                G = NeuronPopulation(4, 'dv/dt=(1-v)/(10*ms):1', threshold='v > 0.1', reset="v=0", method="rk4")
                G.monitor_spike(G.ids)
                connect = Connector(synapse_type="static")
                syn_pp = connect(G, G, G.ids, G.ids, connect=("bernoulli", {"p":0.3}), on_pre='v += 0.1')
                exp.run(5*ms)
            with FileMap("file_analysis.h5") as af:
                with FileMap("file.h5") as f:
                    for run in f.keys():
                        exp_analysis = ExperimentAnalysis(experiment_data=f[run]["data"])
                        exp_analysis.analyze_all()
                        af[run] = exp_analysis.report()

                plotter = ExperimentPlotter(data=f, analysis=af)
                # define plots
                plotter.plot_spike_train()
                # draw plots
                plotter.draw()
                # show plots
                plotter.show()

    """

    def __init__(
        self,
        pop_name_e: str,
        pop_name_i: str,
        data: Union[Dict, persistence.Reader] = None,
        analysis: Union[Dict, persistence.Reader] = None,
        layout: str = "vertical",
        t_start: float = 10.0,
        t_end: float = None,
        **kwargs,
    ):
        if data == None and analysis == None:
            raise ValueError(
                "ExperimentPlotter cannot be instantiated without either data or analysis parameters or both as otw there is nothing to plot."
            )
        if layout not in ["vertical", "horizontal", "quadratic"]:
            raise ValueError(
                f"param layout must be in [vertical, horizontal, quadratic]. Is {layout} "
            )

        self.pop_name_e = pop_name_e
        self.pop_name_i = pop_name_i

        self.data = data
        self.analysis = analysis

        self.plots = []

        self.fig = None
        self.axes = None
        self.fig_title = None
        self.fig_window_title = None

        self.layout = layout

        self.kwargs = kwargs

        # t start, end for data
        self.dt = data["meta"]["dt"]["value"] * 1000
        t = data["meta"]["t"]["value"] * 1000
        self.t_start, self.t_end, self.t_last = compute_time_interval(
            t, self.dt, t_start, t_end
        )

        # t start, end for analysis
        if self.analysis != None:
            t_start_al = self.analysis["meta"]["t_start"]["value"] * 1000
            t_end_al = self.analysis["meta"]["t_end"]["value"] * 1000
            self.t_start_al = (
                0.0 if self.t_start <= t_start_al else self.t_start - t_start_al
            )

            self.t_end_al = None
            if self.t_end != None:
                if t_end_al != None:
                    self.t_end_al = min(self.t_end, t_end_al)
                else:
                    self.t_end_al = self.t_end
            else:
                if t_end_al != None:
                    self.t_end_al = t_end_al
            # # None if self.t_end >= t_end_al else t_end_al - self.t_end
        else:
            self.t_start_al = 0.0
            self.t_end_al = None

    def draw(self):
        num = len(self.plots)

        if self.layout == "quadratic":
            rt = np.sqrt(num)
            rows = int(rt) if num % rt == 0 else int(rt) + 1
            cols = int(rt) if rows * int(rt) >= num else int(rt) + 1
            rows = np.array([int(r) if num % r == 0 else int(r) + 1 for i, r in rt])
            cols = np.array(
                [
                    int(r) if rows[i] * int(r) >= num else int(r) + 1
                    for i, r in enumerate(rt)
                ]
            )

        elif self.layout == "horizontal":
            rows = 1
            cols = num
        elif self.layout == "vertical":
            rows = num
            cols = 1

        self.fig, self.axes = plt.subplots(rows, cols, **self.kwargs)

        self.fig.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.95)

        if isinstance(self.axes, matplotlib.axes.Axes):
            self.plots[0](self.fig, self.axes)
        else:
            for ax, f in zip(self.axes.ravel(), self.plots):
                f(self.fig, ax)
        if self.fig_title != None:
            self.fig.suptitle(self.fig_title)
        if self.fig_window_title != None:
            self.fig.canvas.manager.set_window_title(self.fig_window_title)

    def save(self, path: str):
        plt.savefig(path)

    def show(self):
        plt.show()

    def set_title(self, title):
        self.fig_title = title

    def set_window_title(self, title):
        self.fig_window_title = title

    def plot_spike_train(
        self, pop_name: List[str] = None, color: Dict[str, str] = {}, title: str = ""
    ):
        """
        plot spike train of populations

        :param pop_name: population names of populations tb plotted
        :param color: color by population names
        """
        pop_name = [self.pop_name_e, self.pop_name_i] if pop_name == None else pop_name
        if not all([pn in self.data["SpikeDeviceGroup"].keys() for pn in pop_name]):
            raise ValueError(
                f"Population names {pop_name} not in data['SpikeDeviceGroup'] (parameter provided during __init__)."
            )
        # add plot function
        train = {}
        size = {}
        for pn in pop_name:
            train[pn] = {}
            ids = np.array([])
            spikes = np.array([])
            for i, ts in self.data["SpikeDeviceGroup"][pn]["spike"]["spike_train"][
                "value"
            ].items():
                ts = ts * 1000
                if self.t_start == 0.0 and self.t_end == self.t_last:
                    ts = ts
                else:
                    ts = ts[np.logical_and(ts >= self.t_start, ts <= self.t_end)]

                spikes = np.hstack((spikes, ts))
                ids = np.hstack((ids, np.ones(ts.size) * int(i)))
            train[pn] = (ids, spikes)
            size[pn] = self.data["SpikeDeviceGroup"][pn]["ids"].size

        if self.pop_name_e in pop_name and self.pop_name_e not in color.keys():
            color[self.pop_name_e] = colors["E"]
        if self.pop_name_i in pop_name and self.pop_name_i not in color.keys():
            color[self.pop_name_i] = colors["I"]

        self.plots.append(
            lambda fig, ax: plot_spike_train(
                fig,
                ax,
                pop_name=pop_name,
                train=train,
                size=size,
                color=color,
                title=title,
            )
        )

    def plot_voltage(
        self,
        pop_name: List[str] = [],
        ids: Dict[str, List[int]] = {},
        color: Dict[str, str] = {},
    ):
        """
        plot voltages assuming specific neuron model for E and I populations
        """
        if len(pop_name) == 0:
            pop_name = [self.pop_name_e, self.pop_name_i]
        self.plot_variable(
            ["v" if p != self.pop_name_e else "v_s" for p in pop_name],
            ylabel="voltages [mV]",
            scale=1000,
            pop_name=pop_name,
            ids=ids,
            color=color,
        )

    def plot_variable(
        self,
        variable: Union[str, List[str]],
        ylabel: str,
        scale: float = 1.0,
        pop_name: List[str] = [],
        ids: Dict[str, List[int]] = {},
        color: Dict[str, str] = {},
        title: str = "",
    ):
        """
        :param variable: variable tb plotted or list of variable_names to be mapped to elements of pop_name
        :param ylabel: label of the y axis ( vs time on x axis )
        :param scale: factor by which the variable values provided in __init__() param data are tb scaled
        :param pop_name: names of the populations tb plotted - by default uses __init__() params pop_e and pop_i
        :param ids: mapping of population names to ids for which the variable is tb plotted
        :param color: mapping of population names to colors
        """
        pop_name = [self.pop_name_e, self.pop_name_i] if pop_name == [] else pop_name

        if not isinstance(variable, str) and (
            not isinstance(variable, List) or len(variable) != len(pop_name)
        ):
            raise ValueError(
                "param variable is not of type string and either not of type List (neither) or not of the same length as pop_name"
                + " - note that pop_name is of length 2 by default [self.pop_e,self.pop_i]."
            )

        variable = (
            variable if not isinstance(variable, str) else [variable for _ in pop_name]
        )

        ids = (
            {pn: [0] for pn in pop_name} if isinstance(ids, dict) and ids == {} else ids
        )
        if isinstance(ids, np.ndarray):
            ids = {pn: ids for pn in pop_name}

        if not all(
            [
                pn in self.data["SpikeDeviceGroup"].keys() and pn in ids.keys()
                for pn in pop_name
            ]
        ):
            raise ValueError(
                f"Population names {pop_name} not in data['SpikeDeviceGroup'] (parameter provided during __init__) or not in parameter ids."
            )
        data = self.data["SpikeDeviceGroup"]

        for pn in pop_name:
            if not all(
                [i < len(data[pn]["ids"]) for i in ids[pn]]
                + [i < len(data[pn]["ids"]) for i in ids[pn]]
            ):
                raise ValueError(
                    f"for population {pn}: some ids in ids_e or/and ids_i not contained in data dictionary/Reader passed at initialization"
                )
        values = {}

        for pn, v_var in zip(pop_name, variable):
            idv = np.asarray(ids[pn])

            val = (
                (data[pn]["state"][v_var]["value"][idv] * scale).T
                if not isinstance(data[pn]["state"][v_var], np.ndarray)
                else (data[pn]["state"][v_var][idv] * scale).T
            )
            values[pn] = val

        if not all([values[pn].size == values[pop_name[0]].size for pn in pop_name]):
            raise Exception(
                f"values of param variable: {variable} not all of same length ~ recorded with different time steps."
            )

        # given in seconds s -> ms
        # time is the specific timings during which the variables were measured
        time = data[pop_name[0]]["state"]["t"]["value"] * 1000

        # compute time step btw measurements: v_dt = t_sim / len(timings_v)
        sim_time = self.data["meta"]["t"]["value"] * 1000
        v_dt = sim_time / time.size

        # restrict intervals
        time, _, _ = restrict_to_interval(
            time,
            v_dt,
            self.t_start,
            self.t_end,
        )
        for pn, val in values.items():
            val, _, _ = restrict_to_interval(
                val,
                v_dt,
                self.t_start,
                self.t_end,
            )
            values[pn] = val.T

        color_it = {}
        if self.pop_name_e in pop_name and self.pop_name_e not in color.keys():
            color_it[self.pop_name_e] = color_its["E"](len(ids[pn]))
        if self.pop_name_i in pop_name and self.pop_name_i not in color.keys():
            color_it[self.pop_name_i] = color_its["I"](len(ids[pn]))

        for pop, c_base in color.items():
            color_it[pop] = sequential_palette(c_base, values[pop].size)

        self.plots.append(
            lambda fig, ax: plot_variable(
                fig,
                ax,
                time,
                pop_name,
                ids,
                values,
                color_it,
                "time [ms]",
                ylabel,
                title=title,
            )
        )

    def plot_instantaneous_rate(
        self,
        pop_name: List[str] = [],
        color: Dict[str, str] = {},
        title: str = "",
    ):

        pop_name = [self.pop_name_e, self.pop_name_i] if pop_name == [] else pop_name
        if self.analysis == None or not all(
            [pn in self.analysis["SpikeDeviceGroup"].keys() for pn in pop_name]
        ):
            raise Exception(
                "No analysis object passed for instantiation (__init__) or pop_name_e or pop_name_i not in provided data"
            )

        # note here dt == self.dt as inst rate at sim time step resolution
        dt = self.data["meta"]["dt"]["value"] * 1000

        data = self.analysis["SpikeDeviceGroup"]
        inst_rate = {
            pn: (
                restrict_to_interval(
                    data[pn]["instantaneous_rate"]["value"],
                    dt,
                    self.t_start_al,
                    self.t_end_al,
                )[0],
                data[pn]["instantaneous_rate"]["unit"],
            )
            for pn in pop_name
            if "instantaneous_rate" in data[pn].keys()
        }
        # not that when upper bound given for time we have [t_start, t_end] and therefore (t_end - t_start) // dt + 1
        # whereas no upper bound leaves you with [t_start,:] - the end being determined by the last timestep of brian simulation
        # which simulates timesteps [0.0, sim_time-dt] for a given sim_time ie exactly sim_time // dt steps

        time = np.arange(
            self.analysis["meta"]["t_start"]["value"] * 1000,
            self.analysis["meta"]["t_end"]["value"] * 1000
            if self.t_end_al == None
            else self.analysis["meta"]["t_end"]["value"] * 1000 + dt,
            dt,
        )

        time, _, _ = restrict_to_interval(time, dt, self.t_start_al, self.t_end_al)

        if self.pop_name_e in pop_name and self.pop_name_e not in color.keys():
            color[self.pop_name_e] = colors["E"]
        if self.pop_name_i in pop_name and self.pop_name_i not in color.keys():
            color[self.pop_name_i] = colors["I"]

        self.plots.append(
            lambda fig, ax: plot_instantaneous_rate(
                fig,
                ax,
                time,
                inst_rate,
                color=color,
                title=title,
            )
        )

    def plot_smoothed_rate(
        self,
        pop_name: List[str] = [],
        color: Dict[str, str] = {},
        title: str = "",
        sync_freq: Dict[str, float] = {},
        snr: Dict[str, float] = {},
    ):
        """
        :param sync_freq: synchronization frequency of the population rate by population
        :param snr: signal to noise ratio of the population rate by population
        """

        pop_name = [self.pop_name_e, self.pop_name_i] if pop_name == [] else pop_name

        if len(pop_name) == 0:
            raise ValueError(
                "No population provided -please plot at least one population."
            )

        if self.analysis == None or not all(
            [
                pn in self.analysis["SpikeDeviceGroup"].keys()
                and "instantaneous_rate" in self.analysis["SpikeDeviceGroup"][pn].keys()
                for pn in pop_name
            ]
        ):
            raise Exception(
                "No analysis object passed for instantiation (__init__) or neither pop_name_e nor pop_name_i not in provided data"
                + " or 'instantaneous_rate' not computed."
            )

        # note here dt == self.dt as inst rate at sim time step resolution
        dt = self.data["meta"]["dt"]["value"] * 1000

        rate = {
            pn: (
                restrict_to_interval(
                    data["smoothed_rate"]["value"],
                    dt,
                    self.t_start_al,
                    self.t_end_al,
                )[0],
                data["smoothed_rate"]["mode"][0],
                data["smoothed_rate"]["unit"][0],
            )
            for pn, data in self.analysis["SpikeDeviceGroup"].items()
            if pn in pop_name
        }

        # not that when upper bound given for time we have [t_start, t_end] and therefore (t_end - t_start) // dt + 1
        # whereas no upper bound leaves you with [t_start,:] - the end being determined by the last timestep of brian simulation
        # which simulates timesteps [0.0, sim_time-dt] for a given sim_time ie exactly sim_time // dt steps
        time = np.arange(
            self.analysis["meta"]["t_start"]["value"] * 1000,
            self.analysis["meta"]["t_end"]["value"] * 1000 + dt
            if self.t_end_al != None
            else self.analysis["meta"]["t_end"]["value"] * 1000,
            dt,
        )

        time, _, _ = restrict_to_interval(time, dt, self.t_start_al, self.t_end_al)

        if self.pop_name_e in pop_name and self.pop_name_e not in color.keys():
            color[self.pop_name_e] = colors["E"]
        if self.pop_name_i in pop_name and self.pop_name_i not in color.keys():
            color[self.pop_name_i] = colors["I"]

        self.plots.append(
            lambda fig, ax: plot_smoothed_rate(
                fig,
                ax,
                time,
                rate,
                color=color,
                title=title,
                sync_freq=sync_freq,
                snr=snr,
            )
        )

    def plot_cell_rate(self, pop_name: str, color: Dict[str, str] = {}, title=""):

        if self.analysis == None or not (
            pop_name in self.analysis["SpikeDeviceGroup"].keys()
            and "cell_rate" in self.analysis["SpikeDeviceGroup"][pop_name].keys()
        ):
            raise Exception(
                "No analysis object passed for instantiation (__init__) or pop_name not in 'SpikeDeviceGroup'"
                + " or 'cell_rate' not computed."
            )

        # note here dt == self.dt as inst rate at sim time step resolution
        dt = self.data["meta"]["dt"]["value"] * 1000

        data = self.analysis["SpikeDeviceGroup"][pop_name]

        rate = restrict_to_interval(
            data["cell_rate"]["cell_rate"]["value"],
            dt,
            self.t_start_al,
            self.t_end_al,
        )[0]
        unit = data["cell_rate"]["cell_rate"]["unit"][0]
        ids = data["cell_rate"]["ids"]

        color = {}
        if self.pop_name_e == pop_name and self.pop_name_e not in color.keys():
            color[self.pop_name_e] = colors["E"]
        if self.pop_name_i == pop_name and self.pop_name_i not in color.keys():
            color[self.pop_name_i] = colors["I"]

        self.plots.append(
            lambda fig, ax: plot_cell_rate(
                fig, ax, pop_name, rate, unit, ids, color=color, title=title
            )
        )

    def plot_power_spectrum(self, pop_name: str, title: str = ""):
        """
        Plot log power spectral density against frequencies

        :param pop_name: name of the neuron population (mutually excl w/ pop_name_i)


        self.t_start_al and self.t_end_al are ignored here - as data points used in spectral analysis
        cannot simply be removed if this class defines a narrower bound
        """

        if (
            self.analysis == None
            or pop_name not in self.analysis["SpikeDeviceGroup"].keys()
            or not "psd_complete" in self.analysis["SpikeDeviceGroup"][pop_name]
        ):
            raise Exception(
                f"No analysis object passed for instantiation (__init__) or '{pop_name}' not contained in analysis object or 'psd_complete' computed by ExperimentAnalysis not contained."
            )

        data = self.analysis["SpikeDeviceGroup"][pop_name]
        psd = data["psd_complete"]["psd"]
        freq = data["psd_complete"]["frequency"]["value"]

        color = {}
        if self.pop_name_e == pop_name and self.pop_name_e not in color.keys():
            color[self.pop_name_e] = colors["E"]
        if self.pop_name_i == pop_name and self.pop_name_i not in color.keys():
            color[self.pop_name_i] = colors["I"]

        self.plots.append(
            lambda fig, ax: plot_multitaper_spectrum(
                fig,
                ax,
                psd=psd,
                frequency=freq,
                pop_name=pop_name,
                color=color[pop_name] if color != {} else None,
                title=title,
            )
        )

    def plot_power_spectogram_over_time(self, pop_name: str, title: str = ""):
        """
        Plot power spectral density against frequency and time

        :param pop_name: name of the SpikeDeviceGroup for which the power spectrum is tb plotted

        self.t_start_al and self.t_end_al are ignored here - as data points used in spectral analysis
        cannot simply be removed if this class defines a narrower bound
        """
        if (
            self.analysis == None
            or pop_name not in self.analysis["SpikeDeviceGroup"].keys()
            or not "psd_interval" in self.analysis["SpikeDeviceGroup"][pop_name]
        ):
            raise Exception(
                f"No analysis object passed for instantiation (__init__) or '{pop_name}' not contained in analysis object or 'psd_interval' computed by ExperimentAnalysis not contained."
            )
        data = self.analysis["SpikeDeviceGroup"][pop_name]
        psd = data["psd_interval"]["psd"]
        freq = data["psd_interval"]["frequency"]["value"]
        t_start = self.analysis["meta"]["t_start"]["value"] * 1000
        t_end = self.analysis["meta"]["t_end"]["value"] * 1000

        self.plots.append(
            lambda fig, ax: plot_multitaper_spectogram(
                fig,
                ax,
                psd=psd,
                frequency=freq,
                t_start=t_start,
                t_end=t_end,
                pop_name=pop_name,
                title=title,
            )
        )

    def plot_similarity_distribution(
        self,
        pop_name: str,
        similarity_threshold: np.ndarray,
        rows: int,
        cols: int,
        color: Dict[str, str] = {},
        title="",
    ):
        """
        plot similarity distributions

        :param pop_name: name of the population
        :param similarity_threshold: similarity threshold for respective patterns in key 'similarity_distribution' in parameter analysis passed in :meth:`__init__` (N x 1) -
                        needs tb computed externally
        :param rows: number of rows into which the axes/subplot is subdivided (rows * cols >= #patterns must hold)
        :param cols: number of columns into which the axes/subplot is subdivided (rows * cols >= #patterns must hold)
        """

        if self.analysis == None or not (
            pop_name in self.analysis["SpikeDeviceGroup"].keys()
            and "pattern" in self.analysis["SpikeDeviceGroup"][pop_name].keys()
            # and "pattern" in self.analysis["SpikeDeviceGroup"][pop_name].keys()
            and all(
                [
                    e in self.analysis["SpikeDeviceGroup"][pop_name]["pattern"].keys()
                    for e in ["similarity_distribution", "similarity_threhsold"]
                ]
            )
        ):
            raise Exception(
                "No analysis object passed for instantiation (__init__) or pop_name not in 'SpikeDeviceGroup'"
                + " or 'similarity_distribution' or 'similarity_threshold' not computed."
            )

        similarity_distribution = self._analysis["SpikeDeviceGroup"][pop_name][
            "pattern"
        ]["similarity_distribution"]

        num_patterns = similarity_distribution.shape[0]

        self.plots.append(
            lambda fig, ax: plot_similarity_distributions(
                fig,
                ax,
                pop_name,
                title,
                rows,
                cols,
                similarity_threshold,
                similarity_distribution,
            )
        )

    def plot_total_synaptic_conductance(
        self, pop_name: str, color: Dict[str, str] = {}, title=""
    ):
        """
        plot total synaptic conductance

        :param pop_name: name of the population
        """

        if self.analysis == None or not (
            pop_name in self.analysis["SpikeDeviceGroup"].keys()
            and "pattern" in self.analysis["SpikeDeviceGroup"][pop_name].keys()
            and all(
                [
                    e in self.analysis["SpikeDeviceGroup"][pop_name]["pattern"].keys()
                    for e in ["similarity_distribution", "similarity_threhsold"]
                ]
            )
        ):
            raise Exception(
                "No analysis object passed for instantiation (__init__) or pop_name not in 'SpikeDeviceGroup'"
                + " or 'similarity_distribution' or 'similarity_threshold' not computed."
            )

        total_synaptic_conductance = self._analysis["SpikeDeviceGroup"][pop_name][
            "total_synaptic_conductance"
        ]

        self.plots.append(
            lambda fig, ax: plot_total_synaptic_conductance(
                fig,
                ax,
                pop_name,
                title,
                total_synaptic_conductance,
            )
        )
