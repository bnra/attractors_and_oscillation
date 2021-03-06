import math
from typing import Callable, List, Tuple, Dict, Union, Iterable
import brian2
from brian2 import NeuronGroup, StateMonitor, TimedArray, ms, check_units
from brian2.input.poissongroup import PoissonGroup
from brian2.monitors.ratemonitor import PopulationRateMonitor
from brian2.monitors.spikemonitor import SpikeMonitor
from brian2.synapses.synapses import Synapses
from brian2.units.fundamentalunits import Quantity
from brian2.units.stdunits import Hz
import numpy as np
import os
from abc import ABCMeta, abstractproperty

from utils import (
    Brian2UnitError,
    get_brian2_base_unit,
    convert_and_clean_brian2_quantity,
    retrieve_callers_context,
    retrieve_callers_frame,
    unique_idx,
    unwrap_brian2_variable_view,
)
from connectivity import all2all, bernoulli


class SpikeDeviceGroup(metaclass=ABCMeta):
    """
    Defines Interface for spiking devices and interfaces with :class:`brian2.SpikeMonitor` and :class:`brian2.PopulationRateMonitor`
    to provide monitoring of spike trains and population rates.
    """

    def __init__(self):

        self._rate = None
        self._spike = None

    @abstractproperty
    def _pop(self):
        pass

    @abstractproperty
    def ids(self):
        pass

    @property
    def monitored(self):
        """
        :return: dictionary of recorded variables by :class:`brian2.SpikeMonitor` and :class:`brian2.PopulationRateMonitor`
        """

        data = {}
        data["device"] = {"class": self.__class__.__name__}

        if hasattr(self._spike, "t"):

            states = self._spike.get_states()
            idx, vals = states["i"], states["t"]
            ids, indices = unique_idx(idx)
            vals, unit = convert_and_clean_brian2_quantity(vals)

            data["spike"] = {"spike_train": {}}

            data["spike"]["spike_train"]["value"] = {
                str(i): np.sort(vals[idx]) for i, idx in zip(ids, indices)
            }

            data["spike"]["spike_train"]["unit"] = str(unit)

        if self._rate != None:
            data["rate"] = {}

            rec_clean, unit_str = convert_and_clean_brian2_quantity(
                self._rate.t.variable.get_value_with_unit()
            )
            data["rate"]["t"] = {"value": rec_clean, "unit": unit_str}

            rec_clean, unit_str = convert_and_clean_brian2_quantity(
                self._rate.rate.variable.get_value_with_unit()
            )
            data["rate"]["rate"] = {"value": rec_clean, "unit": unit_str}

            rec_clean, unit_str = convert_and_clean_brian2_quantity(
                self._rate.smooth_rate(window="gaussian", width=1 * ms)
            )
            data["rate"]["smoothed"] = {"value": rec_clean, "unit": unit_str}

        return data

    def monitor_spike(self, ids: List[int], variables: List[str] = []):
        """
        Register neuron ids for monitoring of spikes and related variables of neurons

        :param ids: list of neuron ids that are to be monitored on spike for each neuron
        :param variables: list of neuron variables that are to be monitored additionally to the neuron id for each spike
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

    def monitor_rate(self, **kwargs):
        """
        Register neuron population for rate monitoring
        """
        self._rate = PopulationRateMonitor(self._pop)


class PoissonDeviceGroup(SpikeDeviceGroup):
    """
    Convenience class for interfacing with the :class:`brian2.PoissonGroup` of the poisson devices in the population
    """

    def __init__(self, size: int, rate: Union[Quantity, Callable, str]):

        self.rate_poisson = rate

        self._devices = PoissonGroup(size, rate)

        super().__init__()

    @property
    def _pop(self):
        return self._devices

    @property
    def ids(self):
        """
        ids are unique to a device group and chosen tb equal to the index of a device within the device group - therefore ids start at 0 and are contiguous
        (an index is valid for a given group if index in [0, group.size - 1])
        :return: poisson device ids of the instance of :class:`PoissonDeviceGroup` unique to the instance only! (same as :class:`brian2.PoissonGroup`)
        """
        return [*self._pop.i]

    @property
    def monitored(self):
        data = super().monitored
        data["device"]["rate"] = self.rate_poisson
        return data

    @staticmethod
    @check_units(offset=1, amplitude=1, angularfrequency=1, timeshift=1)
    def create_time_variant_rate(
        offset: float = 1.0,
        amplitude: float = 1.0,
        angularfrequency: float = 2 * np.pi,
        timeshift: float = 0.0,
    ):
        """
        Create a time variant rate to pass to :meth:`PoissonDeviceGroup.__init__()` to create inhomogeneous poisson processes
        rate: [ms]->[kHz]: t -> (offset + cos((t - timeshift[ms]) * angularfrequency[Hz]) * amplitude) * kHz

        :param offset: offset of the rate function
        :param amplitude: scaling factor for amplitude of the rate function
        :param angularfrequency: angular frequency of the rate function [Hz]
        :param timeshift: time shift of the rate function [ms]
        :return:  expression representing the time variant rate function, which specifies the rate in kHz per definition
        """
        return f"({offset} + cos((t - {timeshift} * ms) * {angularfrequency} * Hz) * {amplitude}) * kHz"


class PoissonBlockedStimulus(PoissonDeviceGroup):
    @check_units(
        t=ms,
        stimulus_dt=ms,
    )
    def __init__(
        self,
        size: int,
        pattern: np.ndarray,
        block_interval: List[Tuple[int, int]],
        one_rate: Union[Quantity, str],
        zero_rate: Union[Quantity, str],
        t: Quantity,
        stimulus_dt: Quantity,
    ):
        """
        :param size: size of the group ~ number of spike devices
        :param pattern:  pattern (mask) across all spike devices in the group (shape: (size,) ) - used for setting rate for all indices in block_interval
        :param block_interval: set of half-open time intervals ( [start,end) ) of (time) indices of blocked array tb set to rate
        :param t: simulation time [ms]
        :param stimulus_dt: time step of the stimulus ~ size of one block for which rate is held constant at stimulus_block[i]
        """

        stimulus = self.__class__.create_blocked_rate(
            size, pattern, block_interval, one_rate, zero_rate, t, stimulus_dt
        )
        self.stimulus = stimulus

        super().__init__(size, rate="stim(t,i)")

        # namespace allows adding variables and functions to the resolution process of string equations of class brian2.Group
        # - see https://brian2.readthedocs.io/en/stable/advanced/namespaces.html
        self._devices.namespace["stim"] = stimulus

    @staticmethod
    @check_units(
        t=ms,
        stimulus_dt=ms,
    )
    def create_blocked_rate(
        size: int,
        pattern: np.ndarray,
        block_interval: List[Tuple[int, int]],
        one_rate: Union[Quantity, str],
        zero_rate: Union[Quantity, str],
        t: Quantity,
        stimulus_dt: Quantity,
    ):
        """
        generate an array of rates across devices numbering 'size' and time blocks of length stimulus_dt
        - rates of individual devices are set according to pattern (mask of one devices) and one_rate (rate of one devices)
        and zero_rate (rate of zero devices) across time blocks in the interval block_interval ( [start,end) );
        all rates in time blocks not in the interval block_interval are 0.0

        :param size: size of the group ~ number of spike devices
        :param pattern:  pattern (mask) across all spike devices in the group (shape: (size,) ) - used for setting rate for all indices in block_interval
        :param block_interval: set of half-open time intervals ( [start,end) ) of (time) indices of blocked array tb set to rate
        :param t: simulation time [ms]
        :param stimulus_dt: time step of the stimulus ~ size of one block for which rate is held constant in interval t e [t' * stimulus_dt, (t'+1) * stimulus_dt]: stimulus_block[t]
        :return: rates for individual devices across time blocks of size stimulus_dt
        """
        if t < stimulus_dt:
            raise ValueError(
                f"t >= stimulus_dt must hold otw. ill-defined.Is t:{ t }, stimulus_dt: { stimulus_dt }."
            )
        if size != pattern.shape[0]:
            raise ValueError(
                "group size (param size) and length of first dimension of pattern (param pattern) differ."
            )

        num_blocks = int(np.ceil(t / stimulus_dt))

        if not all(
            [
                len(itv) == 2
                and isinstance(itv[0], int)
                and isinstance(itv[1], int)
                and itv[1] > itv[0]
                and itv[0] >= 0
                and itv[1] <= num_blocks
                for itv in block_interval
            ]
        ):

            errs = [
                [
                    len(itv) == 2,
                    isinstance(itv[0], int),
                    isinstance(itv[1], int),
                    itv[1] > itv[0],
                    itv[0] >= 0,
                    itv[1] <= num_blocks,
                ]
                for itv in block_interval
            ]
            print(errs)
            raise ValueError(
                f"block_interval must be a set of exactly two integer values x,y where y > x, x >= 0 and y < size (param size), where x >= 0, x <=y, y <= number_time_steps, where"
                + f" number time steps is t/dt. Is {block_interval}."
            )

        rate = np.zeros_like(pattern, dtype=float)
        rate[pattern] = one_rate / Hz
        rate[pattern == False] = zero_rate / Hz

        # shape: t,i ~ t rows and i columns ~ one row represents spike rates across devices for one specific time block t
        stimulus_block = np.zeros(num_blocks * size).reshape(num_blocks, size)
        for x, y in block_interval:
            stimulus_block[x:y] = np.tile(rate, y - x).reshape(-1, rate.shape[0])
        return TimedArray(stimulus_block * Hz, dt=stimulus_dt)

    @property
    def monitored(self):
        data = super().monitored
        stim, unit = convert_and_clean_brian2_quantity(self.stimulus)
        data["device"]["stimulus"] = {"value": stim.values, "unit": unit}
        return data

    @staticmethod
    @check_units(
        offset=ms,
        stim_dur=ms,
        stim_relax=ms,
        sim_t=ms,
        stimulus_dt=ms,
    )
    def create_blocked_interval(
        offset: Quantity, stim_dur: Quantity, stim_relax: Quantity, sim_t: Quantity
    ) -> Tuple[List[Tuple[int, int]], Quantity]:
        """
        :param offset: time offset [ms] - offset where no stimulus is presented
        :param stim_dur: stimulus duration [ms] - duration of an instance of stimulus presentation
        :param stim_relax: stimulus relaxation [ms] - relaxation period between stimulus presentations
        :param sim_t: duration of the simulation [ms]
        :return: set of block intervals and stimulation time step
        """

        if (
            offset / ms % 1.0 != 0.0
            or stim_dur / ms % 1.0 != 0.0
            or stim_relax / ms % 1.0 != 0.0
        ):
            raise ValueError(
                "quantities [ms]: offset, stim_dur and stim_relax must be whole numbers (x % 1.0 == 0.0)"
            )

        offset = int(offset / ms)
        stim_dur = int(stim_dur / ms)
        stim_relax = int(stim_relax / ms)
        sim_t = int(sim_t / ms)

        # time step of the blocked interval at which the time dimension of the timed array is resolved - note gcd(a,b,c) = gcd(gcd(a,b),c)
        stimulus_dt = math.gcd(math.gcd(stim_dur, stim_relax), offset)

        def convert_time_to_step(t: float, dt: float):
            return int(np.ceil(t / dt))

        return [
            (
                convert_time_to_step(offset + (stim_dur + stim_relax) * i, stimulus_dt),
                convert_time_to_step(
                    offset + (stim_dur + stim_relax) * (i) + stim_dur, stimulus_dt
                ),
            )
            for i in range(int(np.floor((sim_t - offset) / (stim_dur + stim_relax))))
        ], stimulus_dt * ms


class NeuronPopulation(SpikeDeviceGroup):
    """
    Convenience class for interfacing with the :class:`brian2.NeuronGroup` and the respective :class:`brian2.StatusMonitor` of the neurons in the population


    Example Instantiation of Neuron Population and Initialization of Membrane Potential

    (note if variable is of same dimension as the neuron population use :prop:`NeuronPopulation.size`
    instead of :meth:`NeuronPopulation.get_var_size()`)

    .. testsetup::

        import numpy as np
        from brian2 import mV
        from network import NeuronPopulation
        from distribution import draw_normal
        from BrianExperiment import BrianExperiment

    .. testcode::

        with BrianExperiment():
            N = NeuronPopulation(1000,'dv/dt = (1-v)/tau : volt')
            mu = 0.
            sigma = 1.
            N.set_pop_var("v", draw_normal(mu=mu, sigma=sigma, size=N.get_pop_var_size("v")) * mV)
            # N.set_pop_var("v", draw_normal(mu=mu, sigma=sigma, size=N.size) * mV)
            vals = N.get_pop_var("v") / mV
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"mu:    is w/in 0.1 tolerance ({abs(mean - mu) / sigma < 0.1})")
            print(f"sigma: is w/in 0.1 tolerance ({abs(std - sigma) / sigma < 0.1})")

    .. testoutput::

        mu:    is w/in 0.1 tolerance (True)
        sigma: is w/in 0.1 tolerance (True)

    """

    # add param voltage_init:str=uniform
    def __init__(self, size: int, eqs: str, *args, **kwargs):
        """
        :param size: neuron population size
        :param eqs: eqs used for modelling neuron
        """

        self._population = NeuronGroup(size, eqs, *args, **kwargs)

        self._eqs = eqs

        self._mon = None
        super().__init__()

    def set_pop_var(self, variable: str, value: Union[Quantity, np.ndarray]):
        """
        set population variable - variables defined in eqs param of :meth:`NeuronPopulation.__init__()`

        :param variable: name of variable used in eqs
        :param value: value tb assigned to param variable
        """
        if not variable in self._pop.variables.keys():
            raise ValueError(
                f"No such variable {variable} in neuron population. Available variables: {list(self._pop.variables.keys())}."
            )
        shape_pop = np.array(self.get_pop_var(variable)).shape
        shape_val = np.array(value).shape
        # value needs tb scalar or of same shape as the variable in model
        if not (np.array(value).size == 1 or shape_pop == shape_val):
            raise ValueError(
                f"Parameter value must be of same shape as the neuron population { shape_pop }, but is of shape { shape_val }."
            )

        x = self.get_pop_var(variable)
        if isinstance(x, Quantity):
            base_unit_pop = get_brian2_base_unit(self.get_pop_var(variable))
            base_unit_val = get_brian2_base_unit(value)
            if not base_unit_pop == base_unit_val:
                raise Brian2UnitError(
                    f"Base unit of variable { variable } in population does not match base unit of value {value}, {base_unit_pop} and {base_unit_val} respectively."
                )
        self._pop.variables[variable].set_value(value)

    def get_pop_var(self, variable: str) -> Quantity:
        """
        get population variable - variables defined in eqs param of :meth:`NeuronPopulation.__init__()`

        :param variable: name of variable used in eqs
        :return: value bound to param variable
        """
        return self._pop.variables[variable].get_value_with_unit()

    def get_pop_var_size(self, variable: str) -> int:
        """
        get size of a population variable - variables defined in eqs param of :meth:`NeuronPopulation.__init__()`

        :param variable: name of variable used in eqs
        :return: size of value bound to param variable
        """
        return np.array(self.get_pop_var(variable)).size

    @property
    def _pop(self):
        return self._population

    @property
    def size(self):
        """
        :return: size of the instance of :class:`NeuronPopulation`
        """
        return len(self._pop.i)

    @property
    def ids(self):
        """
        ids are unique to a neuron population and chosen tb equal to the index of a neuron within the population - therefore ids start at 0 and are contiguous
        (an index is valid for a given population if index in [0, pop.size - 1])
        :return: neuron ids of the instance of :class:`NeuronPopulation` unique to the instance only! (same as :class:`brian2.NeuronGroup`)
        """
        return [*self._pop.i]

    @property
    def monitored(self):
        """
        :return: dictionary of recorded variables and their recorded values
        """

        data = super().monitored
        data["device"]["eqs"] = self._eqs
        if hasattr(self._mon, "recorded_variables"):
            data["state"] = {}
            rec_clean, unit_str = convert_and_clean_brian2_quantity(
                self._mon.t.variable.get_value_with_unit()
            )
            data["state"]["t"] = {"value": rec_clean, "unit": unit_str}

            for k in [*self._mon.recorded_variables.keys()]:
                recs = getattr(self._mon, k)
                if isinstance(recs, Quantity):
                    rec_clean, unit_str = convert_and_clean_brian2_quantity(recs)
                    data["state"][k] = {"value": rec_clean, "unit": unit_str}
                else:
                    data["state"][k] = recs

        return data

    def __len__(self):
        return self.size

    def monitor(self, ids: List[int], variables: List[str] = [], dt: float = None):
        """
        Register neuron ids for monitoring of states neuron variables

        :param ids: list of neuron ids whose states are to be monitored for each neuron
        :param variables: list of variables that are to be monitored for each of the neurons
        :param dt: time step to be used for monitoring - df: time step specified in :meth:`BrianExperiment.__init__()`
                   of enclosing instance of :class:`BrianExperiment` used
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
        # valid values see Synapse.monitor (below) - sensible choices: after_thresholds, end
        self._mon = StateMonitor(
            self._pop, variables, record=ids_monitored, when="end", dt=dt
        )



class Synapse:
    """
    Convenience class for interfacing with the created instance of :class:`brian2.synapses.synapses.Synapses` and the respective :class:`brian2.StatusMonitor` of the Synapse
    An instance of this class :class:`Synapse` is returned by :meth:`Connector.__call__()`

    multi-synapses (>1 synpase btw same source and dest) not supported (see member multisynaptic_index of :class:`brian2.synapses.synapses.Synapses`)
    """

    def __init__(
        self,
        synapse_object: Synapses,
        synapse_params: Dict[str, np.ndarray] = {},
        on_pre=None,
    ):
        """
        :param synapse_object: instance which is wrapped by this class
        :param synapse_params: parameters that are set on instance of :class:`brian2.Synapses`
        """

        # retrieve top most stack frame of a function call made to a function that does not reside within this file
        # eg calls from Connector.__call__ will be 'peeled away' as well and the call frame that calls Connector.__call__ will be returned
        frame_info = retrieve_callers_frame(
            lambda fi: os.path.abspath(fi.filename) != os.path.abspath(__file__)
        )

        self._syn_obj = synapse_object

        context = retrieve_callers_context(frame_info)
        self.source = "Failed to find source in scope"
        self.target = "Failed to find target in scope"

        for k, v in context.items():
            if isinstance(v, SpikeDeviceGroup):
                if v._pop == self._syn_obj.source:
                    self.source = {
                        "name": k,
                        "class": v.__class__.__name__,
                        "ids": unwrap_brian2_variable_view(self._syn_obj.i),
                    }
                # can be both - autapse
                if v._pop == self._syn_obj.target:
                    self.target = {
                        "name": k,
                        "class": v.__class__.__name__,
                        "ids": unwrap_brian2_variable_view(self._syn_obj.j),
                    }

        # list of synapse params that are transparent on self
        self._synapse_params = list(synapse_params.keys())

        self.on_pre = on_pre

        # make all parameters in synapse_params on the underlying synapse object transparent

        for p in synapse_params.keys():
            setattr(
                self.__class__,
                p,
                property(
                    lambda self: unwrap_brian2_variable_view(getattr(self._syn_obj, p))
                ),
            )

        self._mon = None

    @property
    def synapse_params(self):
        """
        :return: synapse parameters set on the underlying synapse object
        """
        params = {}
        for p in self._synapse_params:
            v = getattr(self, p)
            if isinstance(v, Quantity):
                vv, unit = convert_and_clean_brian2_quantity(v)
                v = {"value": vv, "unit": unit}
            params[p] = v

        if self.on_pre != None:
            params["on_pre"] = self.on_pre

        return params

    @property
    def synapses(self):
        """
        :return: synapses in terms of tuple of pre- and postsynaptic neuron id (internally resolved to synapse id in :class:`brian2.synapses.synapses.Synapse` only unique to the synapse instance)
        """
        # pre -and postsynaptic neuron of each synapse with index i are stored at index i of member syn_obj.i, syn_obj.j respectively
        # (where syn_obj is an instance of brian2.synapses.synapses.Synapses)
        return np.vstack(
            (self._syn_obj.i, self._syn_obj.j)
        ).T  # list(zip(self._syn_obj.i, self._syn_obj.j))

    @property
    def monitored(self):
        """
        :return: dictionary of recorded variables and their recorded values
        """
        return (
            {
                "state": {
                    k: dict(
                        zip(
                            ("value", "unit"),
                            convert_and_clean_brian2_quantity(getattr(self._mon, k)),
                        )
                    )
                    if isinstance(getattr(self._mon, k), Quantity)
                    else getattr(self._mon, k)
                    for k in [*self._mon.recorded_variables.keys(), "t"]
                }
            }
            if hasattr(self._mon, "recorded_variables")
            else {}
        )

    def __len__(self):
        return len(self.synapses)

    def monitor(
        self,
        synapses: Union[np.ndarray, List[Tuple[int, int]]],
        variables: List[str],
        dt: Quantity = None,
    ):
        """
        Register synapses for monitoring

        :param synapses: group of synapses defined as a tuple of the pre- and postsynaptic neuron ids that are to be monitored
                        - note that if this instance manages a large number of synapses and the number of elements provided
                        is of the same order mapping to brian2 indices will be prohibitively expensive unless param synapses == Synapse.synapses (property)
        :param variables: list of variables that are to be monitored for each of the synapses
        :param dt: time step to be used for monitoring - df: time step specified in :meth:`BrianExperiment.__init__()`
                   of enclosing instance of :class:`BrianExperiment` used
        """
        missing_vars = [v for v in variables if v not in self._syn_obj.variables.keys()]
        if len(missing_vars) > 0:
            raise ValueError(
                "The following vars in parameter variables are not part of the synapse model"
                + f" definition: { missing_vars }(see class Connector for definition)"
            )
        syn_ids = self.synapses

        if not isinstance(synapses, np.ndarray):
            synapses = np.asarray(synapses)

        # determine brian2 indices - np logic used to remove bottleneck

        if synapses.size == syn_ids.size and np.all(synapses == syn_ids):
            ids_monitored = np.arange(syn_ids.shape[0])
        else:
            # treat subset
            # note: that if this instance manages a large number of synapses and the number of elements provided
            #       is of the same order this will be prohibitively expensive unless param synapses == Synapse.synapses (property)
            ids_monitored = np.asarray(
                [
                    np.where(np.sum(syn_ids == s.reshape(-1, 2), axis=1) == 2)[0].item()
                    for s in synapses
                ]
            )

        if synapses.shape[0] != ids_monitored.size:
            raise ValueError("Some synapses contained in param synapses do not exist.")

        # log after synaptic update
        # valid values for when={*, before_*, after_*}, where * in {'start', 'groups', 'thresholds', 'synapses', 'resets', 'end'}
        self._mon = StateMonitor(
            self._syn_obj, variables, record=ids_monitored, when="after_synapses", dt=dt
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
            raise NotImplementedError(f"Synapse type { synapse_type } not implemented.")
        else:
            raise ValueError(
                f"No such synapse type. Available types: static | hebbian."
            )

    def __call__(
        self,
        sourcePop: SpikeDeviceGroup,
        destPop: SpikeDeviceGroup,
        sourceIds: List[int],
        destIds: List[int],
        connect: Union[
            Callable[[List[int], List[int]], List[Tuple[int, int]]],
            Tuple[str, Dict[str, Union[int, float]]],
        ],
        syn_params: Dict[str, Quantity] = {},
        model: str = "",
        on_pre: str = None,
        **kwargs,
    ) -> Synapse:
        """
        Creates synaptic connections between two instances of :class:`NeuronPopulation` of synapse type specified in :meth:`Synapse.__init__()`

        :param sourcePop: instance subclassed from :class:`SpikeDeviceGroup` that contains the subset of presynaptic neurons referenced by ids in parameter sourceIds
        :param destPop: instance subclassed from :class:`SpikeDeviceGroup` that contains the subset of postynaptic neurons referenced by ids in parameter destIds
        :param sourceIds: subset of neuron ids for presynaptic neurons for which synapses are tb created
        :param destIds: subset of neuron ids for postsynaptic neurons for which synapses are tb created
        :param connect: Callable or tuple of specifier ct and params that specify the topology between the two instances of :class:`NeuronPopulation`
                        options for topologies in ct: 'all2all' | 'one2one' | 'bernoulli',
                        note that bernoulli requires param 'p'
        :param syn_params: parameters tb set for the synaptic model on a per synapse basis
        :return: instance of :class:`Synapse` which allows interacting with the synapses created
        """

        source_pop_ids = sourcePop.ids
        dest_pop_ids = destPop.ids
        if not all(
            [sid in source_pop_ids for sid in sourceIds]
            + [did in dest_pop_ids for did in destIds]
        ):
            raise ValueError(
                "Some ids in parameter sourceIds or destIds are not contained in sourcePop or destPop, respectively."
            )

        clean_model_lines = [m.strip() for m in model.split("\n") if len(m.strip()) > 0]
        # if len(clean_model_lines) > 0:
        #    raise ValueError(clean_model_lines,[f"{m.split('=')[0].rstrip('+-/* ') if '=' in m else '/'} or {m.split('=')[1] if len(m.split('=')) > 1 else '/'} or {m.split(':')[0].rstrip(' ') if ':' in m else '/'}" for m in clean_model_lines])
        for p, v in syn_params.items():
            # lhs or rhs or variable definition (for v eg. 'v = s :volt' or 'v :volt')
            #  not ( any([p == m.split("=")[0].rstrip(" ") for m in clean_model_lines if "=" in m]) or any([p in m.split("=")[1] for m in clean_model_lines if len(m.split("=")) > 1]) \
            #    or any([p == m.split(":")[0].rstrip(" ") for m in clean_model_lines if ":" in m]))

            # model strings only: '=', no '(+|*|/|-)=' - '(+|*|/|-)' fine if used as operator, ':' used for defining quantities
            # not that 'w = s : volt' does not allow setting w directly (read only) - only by setting s
            # force explicit definition as 'var : unit'

            if not any(
                [
                    p == m.split(":")[0].rstrip(" ")
                    for m in clean_model_lines
                    if ":" in m
                ]
            ):
                raise ValueError(
                    f"variable {p} in param syn_params is not 'explicitly' defined ('var : unit') in model string. Only variables explicitly defined in model string can be set."
                )

        if model != "":
            kwargs["model"] = model
        if on_pre != None:
            kwargs["on_pre"] = on_pre
        syn = self.synapse_type(sourcePop._pop, destPop._pop, **kwargs)

        if isinstance(connect, tuple):
            ct, params = connect
            if ct == "all2all":
                frm, to = all2all(sourceIds, destIds)
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

                frm, to = bernoulli(sourceIds, destIds, params["p"])
                # only connect if sample contains synapses (note frm.size == to.size)
                if frm.size > 0:
                    syn.connect(i=frm, j=to)

            else:
                raise ValueError(
                    f"No such topology { ct }. Choose from all2all | one2one | bernoulli."
                )
        else:
            frm, to = [np.array(e) for e in zip(*connect(sourceIds, destIds))]
            syn.connect(i=frm, j=to)

        if not all(
            [
                hasattr(syn, p)
                and v.size == len(getattr(syn, p))
                and (
                    not isinstance(v, Quantity)
                    or get_brian2_base_unit(v) == get_brian2_base_unit(getattr(syn, p))
                )
                for p, v in syn_params.items()
            ]
        ):
            raise ValueError(
                "Some of the parameters in syn_params are not set as attributes of the synapse object or their values are not of the same size as the number of synapses "
                + "or they are either not unitless (not a brian2 quantity) or their brian2 unit does not match."
            )

        for p, v in syn_params.items():
            # checks done above
            setattr(syn, p, v.ravel())

        return Synapse(syn, syn_params, on_pre)
