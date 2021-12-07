import itertools
from brian2 import StateMonitor, SpikeMonitor, PopulationRateMonitor, ms, kHz, Hz, mV
from brian2.units.fundamentalunits import get_unit
import numpy as np
import time

from BrianExperiment import BrianExperiment

from test.utils import TestCase
from network import NeuronPopulation, Connector, PoissonDeviceGroup
from differential_equations.neuron_equations import PreEq_AMPA
from differential_equations.neuron_parameters import delay_AMPA
from utils import Brian2UnitError, format_duration_ns
from distribution import draw_normal


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

    def test_get_pop_var_when_called_should_return_current_variable(self):
        N = NeuronPopulation(10,'dv/dt = (1-v)/tau : 1')
        self.assertTrue(np.all(N.get_pop_var("v") == np.zeros(10) * mV))

    def test_set_pop_var_when_called_with_wrong_key_should_raise_value_error(self):
        N = NeuronPopulation(10,'dv/dt = (1-v)/tau : 1')
        with self.assertRaises(ValueError):
            N.set_pop_var("bla", np.zeros(10)* mV)

    def test_set_pop_var_when_called_with_value_quantity_of_wrong_length_should_raise_value_error(self):
        N = NeuronPopulation(10,'dv/dt = (1-v)/tau : 1')
        with self.assertRaises(ValueError):
            N.set_pop_var("v", np.zeros(11)* mV)
    
    def test_set_pop_var_when_called_with_value_quantity_of_wrong_shape_should_raise_value_error(self):
        N = NeuronPopulation(10,'dv/dt = (1-v)/tau : 1')
        with self.assertRaises(ValueError):
            N.set_pop_var("v", np.zeros(10).reshape(2,5)* mV)

    def test_set_pop_var_when_called_should_set_respective_variable(self):
        N = NeuronPopulation(10,'dv/dt = (1-v)/tau : volt')
        value = np.arange(10)* mV
        #raise ValueError(get_unit(N._pop.variables['v'].dim))
        N.set_pop_var("v", value)
        #raise ValueError(f"should: {value}, is {N._pop.variables['v'].get_value_with_unit()}")
        self.assertTrue(np.all(N._pop.variables["v"].get_value_with_unit() == value))

    def test_set_pop_var_when_called_with_value_quantity_of_wrong_unit_should_raise_brian2_unit_error(self):
        N = NeuronPopulation(10,'dv/dt = (1-v)/tau : volt')
        with self.assertRaises(Brian2UnitError):
            N.set_pop_var("v", np.arange(10)* ms)

    def test_set_pop_var_when_called_with_value_quantity_of_other_unit_yet_same_base_unit_should_set_correctly(self):
        N = NeuronPopulation(10,'dv/dt = (1-v)/tau : volt')
        value = np.arange(10)* mV
        N.set_pop_var("v", value)
        self.assertTrue(np.all(N._pop.variables["v"].get_value_with_unit() == value))

    def test_set_pop_var_when_mem_pot_intialized_should_set_mem_pot_appropriately(self):
        N = NeuronPopulation(1000,'dv/dt = (1-v)/tau : volt')
        mu = 0.
        sigma = 1.
        N.set_pop_var("v", draw_normal(mu=mu, sigma=sigma, size=N.get_pop_var_size("v")) * mV)
        
        vals = N.get_pop_var("v") / mV
        mean = np.mean(vals) 
        std = np.std(vals) 

        #raise ValueError(f"{abs(mu-mean)}, {abs(sigma - std) / sigma}")
        self.assertTrue(abs(mu-mean) < 0.1 and abs(sigma - std) / sigma < 0.1)

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
            P = PoissonDeviceGroup(1, rate=1 * kHz)
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

            should = offset * kHz * time_elapsed
            #raise ValueError(should, len(P.monitored["spike"]["spike_train"]["0"]))
            self.assertTrue((abs(len(P.monitored["spike"]["spike_train"]["0"]) - should) / should) < 0.1)


class TestConnector(TestCase):
    def test_when_calling_with_connect_set_to_all2all_should_connect_all_pre_to_all_postsynaptic_neurons(self):
        with BrianExperiment():
            E = NeuronPopulation(10, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
            I = NeuronPopulation(10, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")

            connect = Connector(synapse_type="static")
            S = connect(E, I, E.ids, I.ids, connect=("all2all", {}))

            self.assertEqual(S.synapses, list(itertools.product(E.ids, I.ids)))

    def test_when_calling_with_connect_set_to_one2one_should_connect_each_pre_to_respective_postsynaptic_neurons_at_same_index(self):
        with BrianExperiment():
            E = NeuronPopulation(10, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
            I = NeuronPopulation(10, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")

            connect = Connector(synapse_type="static")
            S = connect(E, I, E.ids, I.ids, connect=("one2one", {}))
            self.assertEqual(S.synapses, list(zip(E.ids, I.ids)))

    def test_when_calling_with_connect_set_to_bernoulli_should_connect_each_pre_to_each_postsynaptic_neuron_with_prob_p(self):
        with BrianExperiment():

            #np.random.seed(0)

            E = NeuronPopulation(1000, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
            I = NeuronPopulation(1000, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
            
            p = 0.3

            connect = Connector(synapse_type="static")
            S = connect(E, I, E.ids, I.ids, connect=("bernoulli", {"p": p}))
            #raise ValueError(f"time elapsed { format_duration_ns(time.time_ns() - tt) }")

            #hack for unsetting random seed - set to current time which is always different
            #np.random.seed(int(time.time()* 1000) % 2**32)

            syns = np.array(S.synapses)
            pres = np.unique(syns[:,0], return_counts=True)[1]
            posts = np.unique(syns[:,1], return_counts=True)[1]

            #raise ValueError(S.synapses)
            #raise ValueError(f"pres: {pres}\nposts:{posts}")
            #raise ValueError(f"pres: {max([ abs(c - p * len(I.ids))/(p*len(I.ids)) for c in pres])}\nposts:{max([ abs(c - p * len(E.ids))/(p*len(E.ids)) for c in posts])}")
            
            #raise ValueError(pres)

            e_size = len(E.ids)
            i_size = len(I.ids)

            pres_diff = abs(pres - p * i_size)/(p*i_size) 
            posts_diff = abs(posts - p * e_size)/(p*e_size)
            # check that each neuron in pop is connected (>1 postsyn. neurons)
            #       and that the counts are within tolerance of 0.1  
            #raise ValueError(f"pres: {max(pres_diff)}, posts: {max(posts_diff)}")
            self.assertTrue(pres.size,e_size)
            self.assertTrue(np.all(pres_diff < 0.2))
            self.assertTrue(posts.size, i_size)
            self.assertTrue(np.all(posts_diff < 0.2))
            

    def test_when_calling_with_connect_set_to_callable_should_connect_as_specified_by_function(self):
        with BrianExperiment():
            E = NeuronPopulation(10, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
            I = NeuronPopulation(10, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")

            # here specifies all2all connectivity
            con = lambda pre, post: list(itertools.product(pre, post))

            connect = Connector(synapse_type="static")
            S = connect(E, I, E.ids, I.ids, connect=con)

            self.assertEqual(S.synapses, con(E.ids, I.ids))