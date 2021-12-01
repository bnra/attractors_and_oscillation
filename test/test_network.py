import itertools
from brian2 import StateMonitor, SpikeMonitor, PopulationRateMonitor, ms, khertz, Hz
import numpy as np

from BrianExperiment import BrianExperiment

from test.utils import TestCase
from network import NeuronPopulation, Connector, PoissonDeviceGroup
from differential_equations.neuron_equations import PreEq_AMPA
from differential_equations.neuron_parameters import delay_AMPA


class TestNeuronPopulation(TestCase):
    
    def test_property_ids_when_called_should_return_ids_of_wrapped_neuron_group(self):
        G = NeuronPopulation(4, 'dv/dt=(1-v)/(10*ms):1', threshold='v > 0.6', reset="v=0", method="rk4") 
        self.assertEqual(G.ids, list(G._pop.i))

    def test_monitor_spike_when_called_should_create_SpikeMonitor(self):
        G = NeuronPopulation(4, 'dv/dt=(1-v)/(10*ms):1', threshold='v > 0.6', reset="v=0", method="rk4")
        G.monitor_spike(G.ids)
        self.assertEqual(G._spike.__class__, SpikeMonitor)

    def test_monitor_rate_when_called_should_create_PopulationRateMonitor(self):
        G = NeuronPopulation(4, 'dv/dt=(1-v)/(10*ms):1', threshold='v > 0.6', reset="v=0", method="rk4")
        G.monitor_rate()
        self.assertEqual(G._rate.__class__, PopulationRateMonitor)

    def test_monitor_when_called_should_create_StateMonitor(self):
        G = NeuronPopulation(4, 'dv/dt=(1-v)/(10*ms):1', threshold='v > 0.6', reset="v=0", method="rk4")
        G.monitor(G.ids, ['v'])
        self.assertEqual(G._mon.__class__, StateMonitor)

    
    def test_monitor_spike_when_experiment_run_should_monitor_spike_train(self):
        with BrianExperiment(persist=True, path="file.h5") as exp:
            G = NeuronPopulation(4, 'dv/dt=(1-v)/(10*ms):1', threshold='v > 0.1', reset="v=0", method="rk4")
            G.monitor_spike(G.ids)
            connect = Connector(synapse_type="static")
            syn_pp = connect(G, G, G.ids, G.ids, connect=("bernoulli", {"p":0.3}), on_pre='v += 0.1')
            exp.run(5*ms)
            self.assertTrue("spike" in G.monitored and list(G.monitored["spike"]["spike_train"].keys()) == [str(e) for e in range(4)])

    def test_monitor_rate_when_experiment_run_should_monitor_population_rate(self):
        with BrianExperiment(persist=True, path="file.h5") as exp:
            G = NeuronPopulation(4, 'dv/dt=(1-v)/(10*ms):1', threshold='v > 0.6', reset="v=0", method="rk4")
            G.monitor_rate()
            connect = Connector(synapse_type="static")
            syn_pp = connect(G, G, G.ids, G.ids, connect=("bernoulli", {"p":0.3}), on_pre='v += 0.1')
            exp.run(5*ms)
            self.assertTrue("rate" in G.monitored and G.monitored["rate"]["rate"].shape[0] == int(5*ms / exp.dt) + 1)

    def test_monitor_when_experiment_run_should_monitor_variables_tb_monitored(self):
        with BrianExperiment(persist=True, path="file.h5") as exp:
            G = NeuronPopulation(4, 'dv/dt=(1-v)/(10*ms):1', threshold='v > 0.6', reset="v=0", method="rk4")
            G.monitor(G.ids, ['v'])
            connect = Connector(synapse_type="static")
            syn_pp = connect(G, G, G.ids, G.ids, connect=("bernoulli", {"p":0.3}), on_pre='v += 0.1')
            exp.run(5*ms)

            self.assertTrue("v" in G.monitored["state"] and G.monitored["state"]["v"].shape[0] == 4 \
                and G.monitored["state"]["v"].shape[1] == int(5*ms / exp.dt) + 1)


class TestSynapses(TestCase):
    def test_property_synapses_when_called_should_return_synapses_defined_by_pre_and_postsynaptic_neuron(self):
        with BrianExperiment():
            x = NeuronPopulation(3, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
            y = NeuronPopulation(3, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")

            connect = Connector(synapse_type="static")
            S = connect(x, y, x.ids, y.ids, connect=("all2all", {}))



            self.assertEqual(S.synapses, list(itertools.product(x.ids, y.ids)))

    def test_attributes_source_name_and_target_name_when_synapse_initialized_should_set_to_name_of_respective_neuron_population(self):
        with BrianExperiment():
            E = NeuronPopulation(3, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
            I = NeuronPopulation(3, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")

            connect = Connector(synapse_type="static")
            S = connect(E, I, E.ids, I.ids, connect=("all2all", {}))

            self.assertTrue(S.source == { "name": "E", "class" : E.__class__.__name__ } and S.target == { "name": "I", "class" : I.__class__.__name__ })


class TestPoissonDeviceGroup(TestCase):

    def test_when_poisson_rate_set_should_evoke_spikes_at_that_rate(self):
        with BrianExperiment(dt=0.1 * ms) as exp:
            P = PoissonDeviceGroup(1, rate=1 * khertz)
            P.monitor_spike(P.ids)
            exp.run(500 * ms)
            self.assertTrue((abs(len(P.monitored["spike"]["spike_train"]["0"]) - 500) / 500) < 0.1)


        
    def test_when_poisson_time_variant_rate_set_should_evoke_spikes_proportional_to_the_integral(self):
        # we are making use of angular_frequency = 2 * pi / 1 s -> Integral_0s^1s == offset * 1s * khz
        # integral of the cosinus component over 2*pi = 0
        with BrianExperiment(dt=0.5*ms) as exp:
            
            offset = 1.0
            time_elapsed = 1000.0 * ms
            rate = PoissonDeviceGroup.create_time_variant_rate(offset=offset, amplitude=1.0, angular_frequency=2*np.pi*Hz)
            P = PoissonDeviceGroup(1, rate=rate)
            P.monitor_spike(P.ids)


            exp.run(time_elapsed)

            should = offset * khertz * time_elapsed
            #raise ValueError(should, len(P.monitored["spike"]["spike_train"]["0"]))
            self.assertTrue((abs(len(P.monitored["spike"]["spike_train"]["0"]) - should) / should) < 0.1)


        