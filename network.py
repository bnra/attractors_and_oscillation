from typing import Callable, List, Tuple, Dict, Union
from brian2 import NeuronGroup, StateMonitor, ms
from brian2.monitors.ratemonitor import PopulationRateMonitor
from brian2.monitors.spikemonitor import SpikeMonitor
from brian2.synapses.synapses import Synapses
import numpy as np
import itertools


from differential_equations.neuron_equations import eqs_P, eqs_I


class NeuronPopulation:
    """
    Convenience class for interfacing with the :class:`brian2.NeuronGroup` and the respective :class:`brian2.StatusMonitor` of the neurons in the population
    """

    def __init__(self, size: int, eqs: str, *args, **kwargs):

        self._pop = NeuronGroup(size, eqs, *args, **kwargs)
        self._mon = None
        self._spike = None
        self._rate = None

    @property
    def ids(self):
        """
        :return: neuron ids of the instance of :class:`NeuronPopulation` unique to the instance only! (same as :class:`brian2.NeuronGroup`)
        """
        return [*self._pop.i]

    @property
    def monitored(self):
        """
        :return: dictionary of recorded variables and their recorded values
        """
        data = {}
        for name, mon in [
            ("state", self._mon),
            ("spike", self._spike),
            ("rate", self._rate),
        ]:
            if mon and hasattr(mon, "recorded_variables"):
                data[name] = {
                    k: getattr(mon, k) for k in [*mon.recorded_variables.keys(), "t"]
                }
        return data

    def monitor(self, ids: List[int], variables: List[str] = []):
        """
        Register neuron ids for monitoring of states neuron variables

        :param ids: list of neuron ids whose states are to be monitored for each neuron
        :param variables: list of variables that are to be monitored for each of the neurons
        """

        missing_vars = [v for v in variables if v not in self._pop.variables.keys()]

        if len(missing_vars) > 0:
            raise ValueError(
                "The following vars in parameter variables are not part of the neuron model"
                + f" definition: { missing_vars }"
            )

        neuron_ids = self.ids
        ids_monitored = [neuron_ids.index(n) for n in ids]

        # log after neurons have update
        # valid values see Synapse.monitor (below)
        self._mon = StateMonitor(
            self._pop, variables, record=ids_monitored, when="after_groups"
        )

        # df records at time when event is evoked

    def monitor_spike(self, ids: List[int], variables: List[str] = []):
        """
        Register neuron ids for monitoring of spikes and related variables of neurons

        :param ids: list of neuron ids that are to be monitored on spike for each neuron
        :param variables: list of neuron variables that are to be monitored additionally to the neuron id for each spike, df:[]
        """
        missing_vars = [v for v in variables if v not in self._pop.variables.keys()]
        if len(missing_vars) > 0:
            raise ValueError(
                "The following vars in parameter variables are not part of the neuron model"
                + f" definition: { missing_vars }"
            )
        neuron_ids = self.ids

        ids_monitored = [neuron_ids.index(n) for n in ids]

        self._spike = SpikeMonitor(self._pop, variables=variables, record=ids_monitored)

    def monitor_rate(self, *args, **kwargs):
        """
        Register neuron population for rate monitoring
        """
        self._rate = PopulationRateMonitor(self._pop)


class EINetwork:
    """
    creates an E-I network with fixed equations and parameters

    kwargs based on AAshqar https://github.com/AAshqar/GammaCoupling/blob/master/Network_utils.py
    """

    def __new__(cls, size_e, size_i):
        """
        :return: a tuple of instances of :class:`NeuronPopulation`: an excitatory and an inhibitory network with fixed equations and parameters
        """
        E = NeuronPopulation(
            size_e, eqs_P, threshold="v_s>-30*mV", refractory=1.3 * ms, method="rk4"
        )
        I = NeuronPopulation(
            size_i, eqs_I, threshold="v_s>-30*mV", refractory=1.3 * ms, method="rk4"
        )
        return E, I


class Synapse:
    """
    Convenience class for interfacing with the created instance of :class:`brian2.synapses.synapses.Synapses` and the respective :class:`brian2.StatusMonitor` of the Synapse
    An instance of this class :class:`Synapse` is returned by :meth:`Connector.__call__()`

    multi-synapses (>1 synpase btw same source and dest) not supported (see member multisynaptic_index of :class:`brian2.synapses.synapses.Synapses`)
    """

    def __init__(self, synapse_object: Synapses):
        """
        :param synapse_object: instance which is wrapped by this class
        """

        self._syn_obj = synapse_object
        self._mon = None

    @property
    def synapses(self):
        """
        :return: synapses in terms of tuple of pre- and postsynaptic neuron id (internally resolved to synapse id in :class:`brian2.synapses.synapses.Synapse` only unique to the synapse instance)
        """
        # pre -and postsynaptic neuron of each synapse with index i are stored at index i of member syn_obj.i, syn_obj.j respectively
        # (where syn_obj is an instance of brian2.synapses.synapses.Synapses)
        return list(zip(self._syn_obj.i, self._syn_obj.j))

    @property
    def monitored(self):
        """
        :return: dictionary of recorded variables and their recorded values
        """
        return {
            "state": {
                k: getattr(self._mon, k)
                for k in [*self._mon.recorded_variables.keys(), "t"]
            }
            if self._mon
            else {}
        }

    def monitor(self, synapses: List[Tuple[int, int]], variables: List[str]):
        """
        Register synapses for monitoring

        :param synapses: list of synapses defined as a tuple of the pre- and postsynaptic neuron ids that are to be monitored
        :param variables: list of variables that are to be monitored for each of the synapses
        """
        missing_vars = [v for v in variables if v not in self._syn_obj.variables.keys()]
        if len(missing_vars) > 0:
            raise ValueError(
                "The following vars in parameter variables are not part of the synapse model"
                + f" definition: { missing_vars }(see class Connector for definition)"
            )
        syn_ids = self.synapses
        ids_monitored = [syn_ids.index(s) for s in synapses]
        # log after synaptic update
        # valid values for when={*, before_*, after_*}, where * in {'start', 'groups', 'thresholds', 'synapses', 'resets', 'end'}
        self._mon = StateMonitor(
            self._syn_obj, variables, record=ids_monitored, when="after_synapses"
        )


class Connector:
    """
    Convenience class for creating synaptic connections - wrapping the instantation and initialization of instances of :class:`brian2.synapses.synapses.Synapses`
    """

    def __init__(self, synapse_type: str = "static"):
        """
        Initialize to type of synapse
        TODO: define these two synapse types
        """
        if synapse_type == "static":
            self.synapse_type = lambda source, dest, **kwargs: Synapses(
                source, dest, **kwargs
            )
        elif synapse_type == "hebbian":
            self.synapse_type = lambda source, dest, **kwargs: Synapses(
                source, dest, **kwargs
            )
        else:
            raise ValueError(
                f"No such synapse type. Available types: static | hebbian."
            )

    def __call__(
        self,
        sourcePop: NeuronPopulation,
        destPop: NeuronPopulation,
        sourceIds: List[int],
        destIds: List[int],
        connect: Union[
            Callable[[List[int], List[int]], List[Tuple[int, int]]],
            Tuple[str, Dict[str, Union[int, float]]],
        ],
        **kwargs,
    ) -> Synapse:
        """
        Creates synaptic connections between two instances of :class:`NeuronPopulation` of synapse type specified in :meth:`Synapse.__init__()`

        :param sourcePop: instance of :class:`NeuronPopulation` that contains the subset of presynaptic neurons referenced by ids in parameter sourceIds
        :param destPop: instance of :class:`NeuronPopulation` that contains the subset of postynaptic neurons referenced by ids in parameter destIds
        :param sourceIds: subset of neuron ids for presynaptic neurons for which synapses are tb created
        :param destIds: subset of neuron ids for postsynaptic neurons for which synapses are tb created
        :param connect: Callable or tuple of specifier ct and params that specify the topology between the two instances of :class:`NeuronPopulation`
                        options for topologies in ct: 'all2all' | 'one2one' | 'bernoulli',
                        note that bernoulli requires param 'p'
        :return: instance of :class:`Synapse` which allows interacting with the synapses created
        """

        if not all(
            [sid in sourcePop.ids for sid in sourceIds]
            + [did in destPop.ids for did in destIds]
        ):
            raise ValueError(
                "Some ids in parameter sourceIds or destIds are not contained in sourcePop or destPop, respectively."
            )

        syn = self.synapse_type(sourcePop._pop, destPop._pop, **kwargs)

        if isinstance(connect, tuple):
            ct, params = connect
            if ct == "all2all":
                frm, to = [
                    np.array(e) for e in zip(*[*itertools.product(sourceIds, destIds)])
                ]
                syn.connect(i=frm, j=to)
            elif ct == "one2one":
                # maps linearly in order of provided ids
                if len(sourceIds) != len(destIds):
                    raise ValueError(
                        f"sourceIds and destIds are of differing length ({len(sourceIds)} vs {len(destIds)})"
                        + ", which is not allowed for param connect=('one2one',{...})."
                    )

                syn.connect(i=sourceIds, j=destIds)
            elif ct == "bernoulli":
                if "p" not in params.keys():
                    raise ValueError(
                        "Specify 'p' in parameter connect=('bernoulli', {'p': x })."
                        + f" Is {params}."
                    )
                syn.connect(i=sourceIds, j=destIds, p=params["p"])
            else:
                raise ValueError(
                    f"No such topology { ct }. Choose from all2all | one2one | bernoulli."
                )
        else:
            frm, to = [np.array(e) for e in zip(*connect(sourceIds, destIds))]
            syn.connect(i=frm, j=to)

        return Synapse(syn)
