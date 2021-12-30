import matplotlib
from typing import Dict, Iterator, Union, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns

import persistence
from utils import compute_time_interval, restrict_to_interval


def sequential_palette(base_color: str, n: int):
    """
    create a generator over sequential colors of a primary hue, base_color

    :param base_color: base color for colormap
    :param n: length of the iterator
    """
    cols = ["Blues", "Greens", "Grays", "Oranges", "Purples", "Reds"]
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


def plot_spike_train(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    spike_train_e: Dict[str, np.ndarray],
    spike_train_i: Dict[str, np.ndarray],
    color_e: str,
    color_i: str,
):
    """
    id to spike train plot for excitatory and inhibitory population

    :param fig: figure instance
    :param ax: axis instance
    :param spike_train_e: spike train of the excitatory population for individual neurons
    :param spike_train_i: spike train of the inhibitory population for individual neurons
    :param color_e: color of excitatory population
    :param color_i: color of inhibitory population
    """
    ids, spikes = list(zip(*sorted(spike_train_e.items(), key=lambda x: int(x[0]))))
    points_e = []
    for i, spike in zip(ids, spikes):
        points_e.append(
            ax.scatter(
                spike,
                np.ones(spike.size) * int(i),
                color=color_e,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
                label="E",
            )
        )

    ids_i, spikes_i = list(zip(*sorted(spike_train_i.items(), key=lambda x: int(x[0]))))
    points_i = []
    for i, spike in zip(ids_i, spikes_i):
        points_i.append(
            ax.scatter(
                spike,
                np.ones(spike.size) * (int(i) + int(ids[-1])),
                color=color_i,
                marker=".",
                s=(72.0 / fig.dpi) ** 2,
                label="I",
            )
        )

    # significantly faster than creating a legend for large datasets
    # data points in yrange 10-90%: 1:4 split: 80/5 = 16 -> [10, 73], [74, 90] -> mid points 42, mid points 82
    # ax.annotate(
    #    "E", (0.985, 0.42), xycoords="axes fraction", color=colors["E"], weight="bold"
    # )
    # ax.annotate(
    #    "I", (0.985, 0.82), xycoords="axes fraction", color=colors["I"], weight="bold"
    # )

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("ids")
    ax.legend([tuple(points_e), tuple(points_i)], ["E", "I"])


def plot_voltages(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    times,
    ids_e,
    ids_i,
    voltages_e,
    voltages_i,
    color_it_e: Iterator,
    color_it_i: Iterator,
):
    """
    plot of voltages of excitatory and inhibitory population

    :param fig: figure instance
    :param ax: axis instance
    :param times: time points
    :param ids_e: ids of neurons in excitatory population which are tb plotted
    :param ids_i: ids of neurons in inhibitory population which are tb plotted
    :param voltages_e: voltages of excitatory population
    :param voltages_i: voltages of inhibitory population
    :param color_it_e: iterator over colors for excitatory population
    :param color_it_i: iterator over colors for inhibitory population
    """
    # dont use res here if anything use sth that ensures the peaks are not lost

    for i, id_ in enumerate(ids_e):
        v = voltages_e[i]

        ax.plot(times, v, label=f"E: {id_}", color=next(color_it_e))
        # ax.set_ylabel("potential [mV]")

    for i, id_ in enumerate(ids_i):
        v = voltages_i[i]
        ax.plot(times, v, label=f"I: {id_}", color=next(color_it_i))
        # ax.set_ylabel("potential [mV]")
    ax.set_ylabel("potential [mV]")
    ax.set_xlabel("time [ms]")
    ax.legend()


def plot_instantaneous_rate(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    times: np.ndarray,
    instantaneous_rate: Dict[str, Tuple[np.ndarray, str]],
    color: str,
):
    """
    plot instantaneous rate of populations

    :param fig: figure instance
    :param ax: axis instance
    :param times: time points
    :param instantaneous_rate: instantaneous population rate over time by population
    :param color: color by population
    """
    # dont use res here if anything use convolution or sth
    for pop_name, (pop_rate, unit) in instantaneous_rate.items():
        ax.plot(times, pop_rate, alpha=0.5, label=pop_name, color=color[pop_name])
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(f"instantaneous rate [{unit}]")
    ax.legend()


def plot_smoothed_rate(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    times: np.ndarray,
    smoothed_rate: Dict[str, Tuple[np.ndarray, str, str]],
    color: Dict[str, str],
):
    """
    plot smooth rate of populations

    :param fig: figure instance
    :param ax: axis instance
    :param times: time points
    :param smoothed_rate: smoothed population rate over time by population
    :param color: color by population

    """
    # dont use res here if anything use convolution or sth
    for pop_name, (rate, mode, unit) in smoothed_rate.items():
        ax.plot(times, rate, label=f"{pop_name} (mode {mode})", color=color[pop_name])
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(f"smooth rate [{unit}]")
    ax.legend()


def plot_multitaper_spectrum(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    frequency,
    psd,
    color: str,
    pop_name: str = None,
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

    """

    ax.plot(frequency, np.log(psd), color=color)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("log power ")

    title = "Power spectral density over the entire time series"
    if pop_name != None:
        title += f" (pop. {pop_name})"

    ax.set_title(title)


def plot_multitaper_spectogram(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    frequency: np.ndarray,
    psd: np.ndarray,
    t_start: float,
    t_end: float,
    pop_name: str = None,
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

    title = "Development of power spectral density of the population rate over time"
    if pop_name != None:
        title += f" (pop. {pop_name})"

    ax.set_title(title)


def plot_synchronization_regimes():
    """
    Synchronization features across the regimes Ing, Bifurcation, Ping
    """
    pass


class ExperimentPlotter:
    """
    Plot data from :class:`BrianExperiment.BrianExperiment` and analysis data from :class:`analysis.ExperimentAnalysis`

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

        self.data = data
        self.analysis = analysis

        self.plots = []

        self.fig = None
        self.axes = None

        self.layout = layout

        self.kwargs = kwargs

        # t start, end for data
        self.dt = data["meta"]["dt"]["value"] * 1000
        t = data["meta"]["t"]["value"] * 1000
        self.t_start, self.t_end, self.t_last = compute_time_interval(
            t, self.dt, t_start, t_end
        )

        # t start, end for analysis
        t_start_al = self.analysis["meta"]["t_start"]["value"] * 1000
        t_end_al = self.analysis["meta"]["t_end"]["value"] * 1000
        self.t_start_al = (
            0.0 if self.t_start <= t_start_al else self.t_start - t_start_al
        )
        self.t_end_al = None if self.t_end >= t_end_al else t_end_al - self.t_end

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
        if isinstance(self.axes, matplotlib.axes.Axes):
            self.plots[0](self.fig, self.axes)
        else:
            for ax, f in zip(self.axes.ravel(), self.plots):
                f(self.fig, ax)

    def save(self, path: str):
        plt.savefig(path)

    def show(self):
        plt.show()

    def plot_spike_train(self, pop_name_e: str, pop_name_i: str):
        """
        plot spike train of E and I population

        """
        if not all(
            [
                pn in self.data["SpikeDeviceGroup"].keys()
                for pn in [pop_name_e, pop_name_i]
            ]
        ):
            raise ValueError(
                f"Population names {pop_name_e} and {pop_name_i} not in data['SpikeDeviceGroup'] (parameter provided during __init__)."
            )
        # add plot function
        train = {}
        for pop_name in [pop_name_e, pop_name_i]:
            train[pop_name] = {}
            for i, ts in self.data["SpikeDeviceGroup"][pop_name]["spike"][
                "spike_train"
            ]["value"].items():
                ts = ts * 1000

                if self.t_start == 0.0 and self.t_end == self.t_last:
                    train[pop_name][i] = ts
                else:
                    train[pop_name][i] = ts[
                        np.logical_and(ts >= self.t_start, ts <= self.t_end)
                    ]

        self.plots.append(
            lambda fig, ax: plot_spike_train(
                fig,
                ax,
                train[pop_name_e],
                train[pop_name_i],
                color_e=colors["E"],
                color_i=colors["I"],
            )
        )

    def plot_voltages(self, pop_name_e: str, pop_name_i: str, ids_e=[0], ids_i=[0]):

        if not all(
            [
                pn in self.data["SpikeDeviceGroup"].keys()
                for pn in [pop_name_e, pop_name_i]
            ]
        ):
            raise ValueError(
                f"Population names {pop_name_e} and {pop_name_i} not in data['SpikeDeviceGroup'] (parameter provided during __init__)."
            )

        data_e = self.data["SpikeDeviceGroup"][pop_name_e]
        data_i = self.data["SpikeDeviceGroup"][pop_name_i]

        if not all(
            [i < len(data_e["ids"]) for i in ids_e]
            + [i < len(data_i["ids"]) for i in ids_i]
        ):
            raise ValueError(
                "some ids in ids_e or/and ids_i not contained in data dictionary/Reader passed at initialization"
            )

        # given in voltage V -> mV
        vol_e, _, _ = restrict_to_interval(
            (data_e["state"]["v_s"]["value"][ids_e] * 1000).T,
            self.dt,
            self.t_start,
            self.t_end,
        )
        voltages_e = vol_e.T

        vol_i, _, _ = restrict_to_interval(
            (data_i["state"]["v"]["value"][ids_i] * 1000).T,
            self.dt,
            self.t_start,
            self.t_end,
        )
        voltages_i = vol_i.T

        # given in seconds s -> ms
        time, _, _ = restrict_to_interval(
            data_e["state"]["t"]["value"] * 1000, self.dt, self.t_start, self.t_end
        )

        self.plots.append(
            lambda fig, ax: plot_voltages(
                fig,
                ax,
                time,
                ids_e,
                ids_i,
                voltages_e,
                voltages_i,
                color_it_e=color_its["E"](len(ids_e)),
                color_it_i=color_its["I"](len(ids_i)),
            )
        )

    def plot_instantaneous_rate(self, pop_name_e, pop_name_i):

        if self.analysis == None or not all(
            [
                pn in self.analysis["SpikeDeviceGroup"].keys()
                for pn in [pop_name_e, pop_name_i]
            ]
        ):
            raise Exception(
                "No analysis object passed for instantiation (__init__) or pop_name_e or pop_name_i not in provided data"
            )

        # note here dt == self.dt as inst rate at sim time step resolution
        dt = self.data["meta"]["dt"]["value"] * 1000

        inst_rate = {
            pop_name: (
                restrict_to_interval(
                    data["instantaneous_rate"]["value"],
                    dt,
                    self.t_start_al,
                    self.t_end_al,
                )[0],
                data["instantaneous_rate"]["unit"],
            )
            for pop_name, data in self.analysis["SpikeDeviceGroup"].items()
            if "instantaneous_rate" in data.keys()
        }
        time = np.arange(
            self.analysis["meta"]["t_start"]["value"] * 1000,
            self.analysis["meta"]["t_end"]["value"] * 1000 + dt,
            dt,
        )

        time, _, _ = restrict_to_interval(time, dt, self.t_start_al, self.t_end_al)

        self.plots.append(
            lambda fig, ax: plot_instantaneous_rate(
                fig,
                ax,
                time,
                inst_rate,
                color={pop_name_e: colors["E"], pop_name_i: colors["I"]},
            )
        )

    def plot_smoothed_rate(self, pop_name_e=None, pop_name_i=None):

        pop_names = [pn for pn in [pop_name_e, pop_name_i] if pn != None]

        if len(pop_names) == 0:
            raise ValueError(
                "No population provided -please plot at least one population."
            )

        if self.analysis == None or not all(
            [
                pn in self.analysis["SpikeDeviceGroup"].keys()
                and "instantaneous_rate" in self.analysis["SpikeDeviceGroup"][pn].keys()
                for pn in pop_names
            ]
        ):
            raise Exception(
                "No analysis object passed for instantiation (__init__) or neither pop_name_e nor pop_name_i not in provided data"
                + " or 'instantaneous_rate' not computed."
            )

        # note here dt == self.dt as inst rate at sim time step resolution
        dt = self.data["meta"]["dt"]["value"] * 1000

        inst_rate = {
            pn: (
                restrict_to_interval(
                    data["smoothed_rate"]["value"],
                    dt,
                    self.t_start_al,
                    self.t_end_al,
                )[0],
                data["smoothed_rate"]["mode"],
                data["smoothed_rate"]["unit"],
            )
            for pn, data in self.analysis["SpikeDeviceGroup"].items()
            if pn in pop_names
        }

        time = np.arange(
            self.analysis["meta"]["t_start"]["value"] * 1000,
            self.analysis["meta"]["t_end"]["value"] * 1000 + dt,
            dt,
        )

        time, _, _ = restrict_to_interval(time, dt, self.t_start_al, self.t_end_al)

        self.plots.append(
            lambda fig, ax: plot_smoothed_rate(
                fig,
                ax,
                time,
                inst_rate,
                color={pop_name_e: colors["E"], pop_name_i: colors["I"]},
            )
        )

    def plot_power_spectrum(self, pop_name: str, is_i: bool = False):
        """
        Plot log power spectral density against frequencies

        :param pop_name: name of the neuron population (mutually excl w/ pop_name_i)
        :param is_i: when set assumes population given is the inhibitory population - df false


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

        color = colors["E"] if not is_i else colors["I"]

        self.plots.append(
            lambda fig, ax: plot_multitaper_spectrum(
                fig, ax, psd=psd, frequency=freq, pop_name=pop_name, color=color
            )
        )

    def plot_power_spectogram_over_time(self, pop_name: str):
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
            )
        )
