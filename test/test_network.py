import itertools
from BrianExperiment import BrianExperiment

from test.utils import TestCase
from network import NeuronPopulation, Connector


class TestNeuronPopulation(TestCase):
    pass


# G = NeuronPopulation(4, 'dv/dt=(1-v)/(10*ms):1', threshold='v > 0.6', reset="v=0", method="rk4") 

# G.monitor_rate()
# assert G._rate.__class__ == PopulationRateMonitor,"PopulationRateMonitor not created"

# G.monitor_spike(G.ids)
# assert G._spike.__class__ == SpikeMonitor, "SpikeMonitor not created"


# G.monitor(G.ids)
# assert G._mon.__class__ == StateMonitor, "StateMonitor not created"

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

            self.assertTrue(S.source_name == "E" and S.target_name == "I")