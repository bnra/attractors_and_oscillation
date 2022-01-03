import numpy as np
from brian2 import ms

from test.utils import TestCase
from BrianExperiment import BrianExperiment
from persistence import FileMap
from network import NeuronPopulation, Connector

from analysis import ExperimentAnalysis, gaussian_smoothing

from differential_equations.neuron_equations import eqs_P, PreEq_AMPA
from differential_equations.neuron_parameters import delay_AMPA
from utils import TestEnv


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
