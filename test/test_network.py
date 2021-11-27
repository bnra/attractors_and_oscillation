from brian2 import PopulationRateMonitor, SpikeMonitor, StateMonitor
from BrianExperiment import BrianExperiment

from test.utils import TestCase
from network import NeuronPopulation

class TestNeuronPopulation(TestCase):
    pass


# G = NeuronPopulation(4, 'dv/dt=(1-v)/(10*ms):1', threshold='v > 0.6', reset="v=0", method="rk4") 

# G.monitor_rate()
# assert G._rate.__class__ == PopulationRateMonitor,"PopulationRateMonitor not created"

# G.monitor_spike(G.ids)
# assert G._spike.__class__ == SpikeMonitor, "SpikeMonitor not created"


# G.monitor(G.ids)
# assert G._mon.__class__ == StateMonitor, "StateMonitor not created"