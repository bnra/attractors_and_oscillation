from brian2.core.network import Network
from brian2.groups.neurongroup import NeuronGroup
from brian2.input.poissongroup import PoissonGroup
from brian2.monitors.ratemonitor import PopulationRateMonitor
from brian2.monitors.spikemonitor import SpikeMonitor
import numpy as np
import os
from typing import Any, List, Tuple, Dict, Union
from types import ModuleType
import importlib
import gc

from brian2.units.fundamentalunits import Quantity
from brian2 import ms, defaultclock, StateMonitor, Synapses

from persistence import FileMap, Node, Writer
import persistence
from utils import generate_sequential_file_name, retrieve_callers_context, retrieve_callers_frame, validate_file_path
from network import NeuronPopulation, PoissonDeviceGroup, SpikeDeviceGroup, Synapse

# _ prevents name from being exported
_Data = Union[np.ndarray, Dict[str, "_Data"]]


class BrianExperiment:
    """
    Implements Context Manager Interface for Brian2 Experiments especially setup and teardown as well as handling persistence
    All data monitored by NeuronPopulation instances is automatically persisted as well as time steps of defaultclock.
    Any additional data can be persisted by adding it to a dict passed to :class:`BrianExperiment`.
    It relies on using the default clock for all network components - we save time array only once.

    It is crucial that all network definitions (instances of :class:`NeuronPopulation`, :class:`Synapse`,...) are bound to a unique name,
    as logic in this class makes use of these names, eg. for persisting.

    Note if neuron equations and parameters reside elsewhere (see :attr:`BrianExperiment.neuron_eq_module`, :attr:`BrianExperiment.neuron_param_module`)
    then update these class variables like so:

    .. code-block:: python

        BrianExperiment.neuron_eq_module = xxx
        BrianExperiment.neuron_param_module = xxx

    Example

    .. testsetup::

        import numpy as np
        from BrianExperiment import BrianExperiment
        from persistence import FileMap
        from utils import TestEnv

    .. testcode::

        with TestEnv():
            with BrianExperiment(persist=True, path="file.h5", object_path="/run_1/data") as exp:
                exp.persist_data["mon"] = np.arange(10)
            with FileMap("file.h5") as f:
                print(f["run_1"]["data"]["persist_data"]["mon"])

    .. testoutput::

        [0 1 2 3 4 5 6 7 8 9]

    """

    neuron_eq_module = "differential_equations.neuron_equations"
    neuron_param_module = "differential_equations.neuron_parameters"

    @staticmethod
    def resolve_module_name(mod: ModuleType) -> str:
        return mod.__name__.lstrip(mod.__package__).lstrip(".")

    class PersistData(dict):
        """
        Dictionary-like class whose setability can be switched on or off
        based on whether persist is set on :class:`BrianExperiment` and whether
        the instance of the class is accessed within the context of :class:`BrianExperiment`

        :param persist: whether or not persist is set on the instance of :class:`BrianExperiment`
        :param exp: instance of :class:`BrianExperiment`
        """

        def __init__(self, persist: bool, exp):
            self.persist = persist
            self.exp = exp

        def __setitem__(self, key, value):
            if not self.persist:
                raise Exception(
                    "persist not set in __init__() therefore data cannot be persisted."
                )
            elif not self.exp._in_context:
                raise Exception(
                    "only setable within context (after __enter__() and before __exit__())"
                )
            else:
                if isinstance(value, dict):
                    super().__setitem__(key, self.__class__(self.persist, self.exp))
                    for k, v in value.items():
                        self[key][k] = v
                else:
                    super().__setitem__(key, value)

        def __repr__(self):
            return f"PD:{super().__repr__()}"

    def __init__(
        self,
        dt: Quantity = 0.01 * ms,
        persist: bool = False,
        path: str = "",
        object_path: str = "/",
        neuron_eqs: List[str] = [],
        neuron_params: List[str] = [],
    ):
        """
        Initialize parameters used in setup and teardown.
        All data monitored by NeuronPopulation instances is automatically persisted as well as time steps of defaultclock.
        To add additional data set it on dictionary-like :attr:`BrianExperiment.persist_data`.

        :param dt: timestep of defaultclock
        :param persist: specifies whether data is tb persisted
        :param path: path to the file to be used for persisting - df:None, if not set and persist=True will automatically generate file name 'experiments/experiment_i'
        :param object_path: path within the hdf5 file starting with root '/', eg. '/run_x/data', df:'/'
        :param neuron_eqs: neuron equations to persist - verified against :attr:`BrianExperiment.neuron_eq_module`
        :param neuron_params: neuron parameters to persist - verified against :attr:`BrianExperiment.neuron_param_module`
        """

        if persist:

            if path:
                error = validate_file_path(path, ext=".h5")
                if error:
                    raise ValueError(error)
            else:
                base_dir = os.path.join(os.path.abspath("."), "experiments")
                path = generate_sequential_file_name(base_dir, "exp", ".h5")
                if not os.path.isdir(base_dir):
                    os.makedirs(base_dir)

            error = persistence.opath.verify(object_path, path_type="abs_path")
            if error:
                raise persistence.opath.OpathError(error)

            if neuron_eqs:
                module = importlib.import_module(BrianExperiment.neuron_eq_module)
                missing_names = [
                    eq for eq in neuron_eqs if eq not in module.__dict__.keys()
                ]
                if len(missing_names) > 0:
                    raise ValueError(
                        f"Some equations do not exist within {BrianExperiment.neuron_eq_module}: {missing_names}"
                    )
            if neuron_params:
                module = importlib.import_module(BrianExperiment.neuron_param_module)
                missing_names = [
                    eq for eq in neuron_params if eq not in module.__dict__.keys()
                ]
                if len(missing_names) > 0:
                    raise ValueError(
                        f"Some equations do not exist within {BrianExperiment.neuron_param_module}: {missing_names}"
                    )

        if (
            path != "" or object_path != "/" or neuron_eqs != [] or neuron_params != []
        ) and not persist:
            raise ValueError(
                "A dictionary for param persist_data or/and a path for persistence or/and an object_path or/adn neuron_eqs tb persisted"
                + " or/and neuron_params tb persisted were passed but persist is not set. Please set persist=True to persist data."
            )

        self._persist = persist
        self._dt = dt
        self._path = os.path.abspath(path) if path else None
        self._opath = object_path
        self._neuron_equations = neuron_eqs
        self._neuron_parameters = neuron_params
        self._network = None

        # whether user is within context
        self._in_context = False

        self._device_context = []

        self._persist_data = BrianExperiment.PersistData(persist, self)

    @property
    def persist_data(self):
        """
        special dictionary (:class:`BrianExperiment.PersistData`) that may be populated within the context and whose entries will be persisted on exit if persist is set
        """
        return self._persist_data

    @property
    def path(self):
        """
        path to underlying h5 file - especially useful when no path passed in :meth:`BrianExperiment.__init__()` and it is autogenerated
        """
        return self._path

    @property
    def dt(self):
        """
        timestep to be used in simulation
        """
        return self._dt

    def _retrieve_callers_frame(self):
        # find first caller in call stack who is not in class BrianExperiment
        # note technically we are testing whether method name defined by class and file path is same as file where class is defined
        #      so defining a function with same name as method of class within same file would
        #      break this (counted as class method and therefore ignored)
        return retrieve_callers_frame(lambda fi: not (fi.function in dir(self.__class__) and fi.filename == os.path.abspath(__file__)))
 

    def _retrieve_callers_context(self):
        # retrieve the context: globals updated with locals (ie locals shadow globals if same key in both)
        frame_info = self._retrieve_callers_frame()
        return retrieve_callers_context(frame_info)

    def _save_context(self):
        self._device_context = self._collect_devices()


    def _collect_devices(self):

        brian_devices = []
        # collect all brian2 network devices that were not in scope when the context was entered
        for obj in [
            v
            for v in self._retrieve_callers_context().values()
            if v.__class__ == NeuronPopulation or v.__class__ == Synapse or v.__class__ == PoissonDeviceGroup
        ]:
            # get all StateMonitors, NeuronGroups, Synapses etc. (add Poisson)
            devices = [
                v
                for v in obj.__dict__.values()
                if (v.__class__ == NeuronGroup
                or v.__class__ == PoissonGroup
                or v.__class__ == Synapses
                or v.__class__ == StateMonitor
                or v.__class__ == SpikeMonitor
                or v.__class__ == PopulationRateMonitor)
                and v not in self._device_context
            ]
            for device in devices:
                brian_devices.append(device)
        return brian_devices

    def _reset_context(self):
        """
        delete the underlying brian2 objects of the wrapper classes defined within the context
        of the experiment
        """
        for e in self._collect_devices():
            del e
        gc.collect()

    def _get_namespace(self):
        def clean_ns(ns: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in ns.items() if not k.startswith("__")}

        neuron_parameters_mod = importlib.import_module(
            BrianExperiment.neuron_param_module
        )
        namespace = clean_ns(neuron_parameters_mod.__dict__)
        namespace.update(clean_ns(self._retrieve_callers_context()))
        return namespace

    def run(self, time: Quantity = 0.01 * ms):

        if not self._in_context:
            raise Exception(
                "Call only from within the context: with {self.__class__.__name__}() as exp:\n     exp.run()"
            )

        for device in self._collect_devices():
            self._network.add(device)

        self._network.run(time, namespace=self._get_namespace())

    def __enter__(self):

        # save local context to restore on exit and clear local context added within the context
        self._save_context()


        # all objects created before this call are no longer 'magically' included by run
        defaultclock.dt = self._dt

        # create network and register with class variable
        self._network = Network()

        self._in_context = True

        return self

    @staticmethod
    def _destructure_persist(items: List[Tuple[Writer, _Data]]):
        fm, data = items.pop(0)
        for k in data.keys():
            if isinstance(data[k], dict):
                fm[k] = Node()
                items.append((fm[k], data[k]))
            else:  # np.ndarray
                fm[k] = data[k]

        if len(items) > 0:
            BrianExperiment._destructure_persist(items)

    def __exit__(self, exc_type, exc_value, traceback):

        if self._persist:
            flmp = None
            if os.path.isfile(self._path):
                flmp = FileMap(self._path, mode="modify", object_path=self._opath)
            else:
                flmp = FileMap(self._path, mode="write", object_path=self._opath)
            with flmp as fm:
                # persist all data in the persist_data dictionary
                if self.persist_data:
                    fm["persist_data"] = self.persist_data

                # persist all Neuronpopulations

                fm[SpikeDeviceGroup.__name__] = Node()
                neurp = fm[SpikeDeviceGroup.__name__]
                for i, (k, v) in enumerate(
                    [
                        (k, v)
                        for k, v in self._retrieve_callers_context().items()
                        if isinstance(v, SpikeDeviceGroup)
                    ]
                ):
                    neurp[k] = Node()
                    neurp[k]["ids"] = v.ids
                    mon_data = v.monitored

                    for mon in mon_data.keys():
                        neurp[k][mon] = Node()
                        for var, val in mon_data[mon].items():
                            # note that np.array on ndarrays is idempotent
                            neurp[k][mon][var] = val

                # persist all synapses
                fm[Synapse.__name__] = Node()
                sn = fm[Synapse.__name__]
                for k, v in [
                    (k, v)
                    for k, v in self._retrieve_callers_context().items()
                    if v.__class__ == Synapse
                ]:
                    sn[k] = Node()
                    sn[k]["source"] = v.source
                    sn[k]["target"] = v.target
                    sn[k]["ids"] = np.array(v.synapses)
                    mon_data = v.monitored
                    for mon in mon_data.keys():
                        sn[k][mon] = Node()
                        for var, val in mon_data[mon].items():
                            sn[k][mon][var] = np.array(val)

                # persist all equations passed in __init__()
                if self._neuron_equations:
                    module = importlib.import_module(BrianExperiment.neuron_eq_module)
                    mod_name = BrianExperiment.resolve_module_name(module)
                    fm[mod_name] = Node()
                    for eq in self._neuron_equations:
                        fm[mod_name][eq] = module.__dict__[eq]

                # persist all parameters passed in __init__()
                if self._neuron_parameters:
                    module = importlib.import_module(
                        BrianExperiment.neuron_param_module
                    )
                    mod_name = BrianExperiment.resolve_module_name(module)
                    fm[mod_name] = Node()
                    for param in self._neuron_parameters:
                        fm[mod_name][param] = module.__dict__[param]

        self._in_context = False

        if exc_type != None:
            raise exc_type(exc_value, traceback)
