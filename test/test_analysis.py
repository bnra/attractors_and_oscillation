import numpy as np
from brian2 import ms

from test.utils import TestCase
from BrianExperiment import BrianExperiment
from persistence import FileMap
from network import NeuronPopulation, Connector

from analysis import ExperimentAnalysis, gaussian_smoothing
import analysis

from differential_equations.neuron_equations import eqs_P, PreEq_AMPA
from differential_equations.neuron_parameters import delay_AMPA
from utils import TestEnv


class TestCellRateFromSpikeTrain(TestCase):
    def test_when_passing_only_spikes_outside_time_interval_should_return_rate_zero(
        self,
    ):
        dt = 0.1
        sim_time = 1000.0
        t_start = 200.0
        t_end = 800.0

        spike_train = {"0": None, "1": None, "2": None}
        ids = np.array([0, 1, 2])

        for k in spike_train.keys():
            pre_start_timings = np.random.choice(np.arange(0.0, t_start, dt), size=10)
            post_end_timings = np.random.choice(
                np.arange(t_end + dt, sim_time, dt), size=10
            )
            spike_train[k] = np.hstack((pre_start_timings, post_end_timings))

        _, cell_rate = analysis.cell_rate_from_spike_train(
            t_start, t_end, ids, spike_train
        )

        self.assertTrue(np.all([cell_rate == 0.0]))

    def test_when_passing_spikes_should_compute_rate_per_neuron(self):
        dt = 0.1
        sim_time = 1000.0
        t_start = 200.0
        t_end = 800.0

        spike_train = {"0": None, "1": None, "2": None}
        ids = np.array([0, 1, 2])

        for k in spike_train.keys():
            spike_train[k] = np.random.choice(
                np.arange(t_start, t_end + dt, dt), size=int(k) * 10
            )

        ids_rate, cell_rate = analysis.cell_rate_from_spike_train(
            t_start, t_end, ids, spike_train
        )

        self.assertTrue(
            all(
                [
                    v == int(k) * 10 / (t_end - t_start) * 1000
                    for k, v in zip(ids_rate, cell_rate)
                ]
            )
        )


class TestFctSnr(TestCase):
    @staticmethod
    def psd(x, dt, restrict_to_freq: float = 1000):
        spect = np.fft.fft(x)
        power = np.abs(spect) ** 2 / x.shape[0]
        # freq returned by fftfreq is dimensionless rescale with 1/dt (sampling interval)
        #  as df = 1/T with T = N/dt
        freq = np.fft.fftfreq(x.shape[0]) / dt
        idx = np.logical_and(freq >= 0, freq <= restrict_to_freq)
        return power[idx], freq[idx]

    def test_when_computing_the_snr_of_a_sinusoid_should_return_extremely_high_value(
        self,
    ):
        dt = 0.0001
        t = np.arange(0, 1, dt)
        # sin in the frequency domain only at its frequency != 0 - we should get a very high value
        x = np.sin(t * 2 * np.pi * 10)
        power, freq = TestFctSnr.psd(x, dt)
        snr = analysis.snr(power, freq)

        self.assertTrue(snr >= 2.5e3 / 1e-25)

    def test_when_computing_the_snr_of_a_dirac_should_return_value_proportional_to_half_size_of_bin_size_relative_to_spectrum_remainder(
        self,
    ):
        # value_proportional_to_size_of_bin_size_relative_to_spectrum_remainder
        # for dirac pulse value is proportional to half the bin size relative to spectrum remainder (as argmax will choose first index)
        dt = 0.0001
        t = np.arange(0, 1, dt)

        num_freqs = 100
        bin_size = 10
        # sin in the frequency domain only at its frequency != 0 - we should get a very high value
        x = np.zeros_like(t)
        x[t.shape[0] // 2] = 10.0
        power, freq = TestFctSnr.psd(x, dt, restrict_to_freq=num_freqs)
        snr = analysis.snr(power, freq)
        # half the bin size as argmax is 0 and index 0 itself (+1)
        self.assertTrue(
            snr
            - (bin_size // 2 + 1) / (num_freqs - bin_size) * (num_freqs + 1) * power[0]
            < 0.01
        )


class TestClassExperimentAnalaysis(TestCase):
    def test_instaneous_rate_when_spikes_occur_rate_should_be_a_multiple_of_count_with_factor_dependent_on_dt_and_pop_size(
        self,
    ):
        with TestEnv():
            f_name = "file.h5"
            with BrianExperiment(persist=True, path=f_name) as exp:

                pop_size = 5

                E = NeuronPopulation(
                    pop_size,
                    eqs_P,
                    threshold="v_s>-30*mV",
                    refractory=1.3 * ms,
                    method="rk4",
                )
                E.monitor_rate()
                E.monitor(E.ids, ["v_s"])
                E.monitor_spike(E.ids)

                connect = Connector(synapse_type="static")
                syn_pp = connect(
                    E,
                    E,
                    E.ids,
                    E.ids,
                    connect=("bernoulli", {"p": 0.5}),
                    on_pre=PreEq_AMPA,
                    delay=delay_AMPA,
                )
                syn_pp.monitor(syn_pp.synapses, ["x_AMPA"])

                exp.run(25 * ms)

            with FileMap(path=f_name, mode="read") as f:
                analyzer = ExperimentAnalysis(experiment_data=f, t_start=0.0)
                analyzer.analyze_instantaneous_rate()
                analysis = analyzer.report

                trains = f["SpikeDeviceGroup"]["E"]["spike"]["spike_train"]["value"]
                spikes = np.hstack(list(trains.values())) * 1000

                t = f["meta"]["t"]["value"] * 1000
                dt = f["meta"]["dt"]["value"] * 1000

            rate = analysis["SpikeDeviceGroup"]["E"]["instantaneous_rate"]["value"]

            # timing in ms
            vals, counts = np.unique(spikes, return_counts=True)
            idx_vals = np.asarray(np.ceil(vals / dt), dtype=int)

            spike_counts = np.zeros(int(np.ceil(t / dt)))
            spike_counts[idx_vals] = counts

            self.assertTrue(np.allclose(spike_counts / pop_size * 1000 / dt, rate))


class TestFunctionGaussianSmoothing(TestCase):
    def test_gaussian_smoothing_when_computing_smooth_rate_should_yield_same_result_as_brian2_smoothing_fct(
        self,
    ):

        # brian2 reports differing instantaneous rate and smooth_rate is a method therefore the relevant code is condensed here
        # (see https://brian2.readthedocs.io/en/stable/_modules/brian2/monitors/ratemonitor.html#PopulationRateMonitor)
        def brian2_reference_gaussian(rate: np.ndarray, window_size: float):
            width_dt = int(np.round(2 * window_size / dt))
            x = np.arange(-width_dt, width_dt + 1)
            window = np.exp(-(x ** 2) / (2 * (window_size / dt) ** 2))
            return np.convolve(rate, window / np.sum(window), mode="same")

        with TestEnv():
            f_name = "file.h5"
            with BrianExperiment(persist=True, path=f_name) as exp:

                pop_size = 5

                E = NeuronPopulation(
                    pop_size,
                    eqs_P,
                    threshold="v_s>-30*mV",
                    refractory=1.3 * ms,
                    method="rk4",
                )
                E.monitor_rate()
                E.monitor(E.ids, ["v_s"])
                E.monitor_spike(E.ids)

                connect = Connector(synapse_type="static")
                syn_pp = connect(
                    E,
                    E,
                    E.ids,
                    E.ids,
                    connect=("bernoulli", {"p": 0.5}),
                    on_pre=PreEq_AMPA,
                    delay=delay_AMPA,
                )
                syn_pp.monitor(syn_pp.synapses, ["x_AMPA"])

                exp.run(25 * ms)

            with FileMap(path=f_name, mode="read") as f:
                analyzer = ExperimentAnalysis(experiment_data=f)
                analyzer.analyze_instantaneous_rate()
                analysis = analyzer.report

                t = f["meta"]["t"]["value"] * 1000
                dt = f["meta"]["dt"]["value"] * 1000

            inst_rate = analysis["SpikeDeviceGroup"]["E"]["instantaneous_rate"]["value"]

            computed_rate = gaussian_smoothing(
                instantaneous_rate=inst_rate,
                window_size=2.0,
                one_sigma_window=1.0,
                dt=dt,
            )

            reference_rate = brian2_reference_gaussian(rate=inst_rate, window_size=1.0)

            self.assertTrue(np.allclose(computed_rate, reference_rate))
