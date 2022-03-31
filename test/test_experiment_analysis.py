import numpy as np
from brian2 import ms

from test.utils import TestCase
from BrianExperiment import BrianExperiment
from persistence import FileMap
from network import NeuronPopulation, Connector

from ExperimentAnalysis import ExperimentAnalysis

from utils import TestEnv


from differential_equations.eif_equations import (
    eq_eif_E,
    eq_eif_I,
    pre_eif_E,
    pre_eif_I,
    pre_eif_Pois,
)

from differential_equations.eif_parameters import (
    C_E,
    C_I,
    gL_E,
    gL_I,
    eL_E,
    eL_I,
    deltaT,
    VT,
    V_thr,
    V_r,
    gsynE_E,
    gsynI_E,
    gsynE_I,
    gsynI_I,
    esynE,
    esynI,
    rise_AMPA,
    rise_GABA,
    refractory_E,
    refractory_I,
    decay_AMPA,
    decay_GABA,
    latency_AMPA,
    latency_GABA,
    psx_AMPA,
    psx_GABA,
    psx_AMPA_ext,
    alpha,
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
                    eq_eif_E,
                    threshold="V>V_thr",
                    reset="V = V_r",
                    refractory=refractory_E,
                    method="rk2",
                )

                E.monitor_rate()
                E.monitor(E.ids, ["V"])
                E.monitor_spike(E.ids)


                connect = Connector(synapse_type="static")
                syn_pp = connect(
                    E,
                    E,
                    E.ids,
                    E.ids,
                    connect=("bernoulli", {"p": 0.5}),
                    on_pre=pre_eif_E,
                    delay=latency_AMPA,
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
