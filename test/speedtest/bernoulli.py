from test.utils import SpeedTest
from connectivity import bernoulli, all2all
from network import NeuronPopulation
from brian2 import Synapses

class BrianBernoulli(SpeedTest):
    
    trials = 3
    iterations = 5

    def run(self):
        p = 0.3

        E = NeuronPopulation(1000, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
        I = NeuronPopulation(1000, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")


        syn = Synapses(E._pop,I._pop)
        
        frm, to = all2all(E.ids, I.ids)
        syn.connect(i=frm, j=to, p=p)





    


class ConnectivityBernoulli(SpeedTest):

    trials = 3
    iterations = 5

    def run(self):
        p = 0.3

        E = NeuronPopulation(1000, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
        I = NeuronPopulation(1000, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
        
        frm, to = bernoulli(E.ids, I.ids, p)
        syn = Synapses(E._pop,I._pop)
        # only connect if sample contains synapses (note frm.size == to.size)
        if frm.size > 0:
            syn.connect(i=frm, j=to)