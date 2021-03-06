import string
import numpy as np
import os
from typing import Any, List, Tuple, Dict, Union
from types import ModuleType
import importlib
import gc
import time
from tqdm import tqdm

import brian2
from brian2.units.fundamentalunits import Quantity
from brian2 import ms, defaultclock, StateMonitor, Synapses
from brian2.core.network import Network
from brian2.groups.neurongroup import NeuronGroup
from brian2.input.poissongroup import PoissonGroup
from brian2.monitors.ratemonitor import PopulationRateMonitor
from brian2.monitors.spikemonitor import SpikeMonitor

from persistence import FileMap, Node, Writer
import persistence
from utils import (
    clean_brian2_quantity,
    convert_and_clean_brian2_quantity,
    generate_sequential_file_name,
    format_duration_ns,
    retrieve_callers_context,
    retrieve_callers_frame,
    validate_file_path,
)
from network import NeuronPopulation, PoissonDeviceGroup, SpikeDeviceGroup, Synapse


class TqdmCallBack(tqdm):
    """Provide progress bar updatable via callback based on :class:`tqdm` via :meth:`tqdm.update()`"""

    def __init__(self, report_freq: Quantity, *args, **kwargs):
        """
        :param report_freq: frequency [ms] at which to report progress during the simulation stage - throws exception if not specified in unit :attr:`brian2.ms`
        """
        self.w = None
        self.dec = 3
        if (
            not isinstance(report_freq, Quantity)
            or clean_brian2_quantity(report_freq)[1] != "ms"
        ):
            raise ValueError(
                "param report_freq must be of unit brian2.ms and type brian2.Quantity"
            )
        self.report_freq, _ = clean_brian2_quantity(report_freq)
        # cut off decimals - show decimal if not log(x) % 1.0 == 0.0
        rep_dec = int(np.log10(self.report_freq))
        self.report_dec = 3 - rep_dec if rep_dec <= 3 else 0
        self.report_w = 3 + self.report_dec
        super().__init__(*args, **kwargs)

    def update_cb(
        self, elapsed: Quantity, completed: float, start: Quantity, duration: Quantity
    ):
        """
        update progress bar

        :param elapsed: total real time since start of the experiment
        :param completed: fraction in [0,1] indicating completion
        :param start: start of the experiment in biological time
        :param duration: total duration of the experiment in biological time
        """

        elapsed = convert_and_clean_brian2_quantity(elapsed)[0]
        start_ms = convert_and_clean_brian2_quantity(start)[0] * 1000
        duration_ms = convert_and_clean_brian2_quantity(duration)[0] * 1000

        if not self.w:
            w = np.log10(duration_ms)
            self.w = int(w) + self.dec if w % 1.0 == 0.0 else int(w) + self.dec + 1

        bio_progress = round(duration_ms * completed, 3)
        # bio_end_time = duration_ms * completed if completed > 1.0 else duration_ms

        bio_end_time_offset = start_ms + duration_ms  # bio_end_time
        # completed is not properly clamped to [0,1]
        # completed = max(1.0, min(0.0, completed))

        # print(type(elapsed_ms), type(start_ms), type(duration_ms))
        self.set_description(
            f"real time elapsed [s]: {elapsed:{self.report_w}.{self.report_dec}f}  |  biological time [ms]: {start_ms:{self.w}.{self.dec}f}"
            + f" -> {bio_progress:{self.w}.{self.dec}f} ---> {bio_end_time_offset:{self.w}.{self.dec}f}  |  "
        )
        self.total = duration_ms
        return self.update(bio_progress - self.n)


class TimeTracker:
    """
    track time durations of sequential stages of a process
    """

    def __init__(self, verbose: bool = False):
        """
        :param verbose: whether to notify on beginning and ending of each of the stages (including duration) - prints to stdout
        """
        self.last_stage = None
        self._timings = {}
        self.last_stamp = None
        self._verbose = verbose

    def add_timing(self, process: str):
        """
        add a new stage - this marks the end of the previous stage if it exists
        """
        tt = time.time_ns()

        if self.last_stage:
            self._timings[self.last_stage] = tt - self.last_stamp

            if self.verbose:
                print(
                    f"Stage {self.last_stage} done.\n({format_duration_ns(self._timings[self.last_stage])})"
                )
        if self.verbose:
            print(f"\nStarting stage {process}\n...")

        self.last_stage = process
        self.last_stamp = tt

    def finalize(self):
        """
        end the previous stage without beginnning a new stage
        """
        if not self.last_stage:
            raise Exception(
                "You cannot call finalize() when you haven't called add_timing() since the last call to finalize()"
            )
        tt = time.time_ns()
        self._timings[self.last_stage] = tt - self.last_stamp

        if self.verbose:
            print(
                f"Stage {self.last_stage} done.\n({format_duration_ns(self._timings[self.last_stage])})"
            )

        self.last_stage = None
        self.last_stamp = None

    @property
    def timings(self):
        """
        timings of all tracked and ended processes
        """
        return self._timings

    @property
    def verbose(self):
        """
        indicates whether the TimeTracker is used in verbose mode, where TimeTracker will print progress
        (verbose is set in :class:`TimeTracker.__init__()`)
        """
        return self._verbose


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

    Note if neuron equations and parameters reside elsewhere (see :attr:`neuron_eq_module`, :attr:`neuron_param_module`)
    then pass the corresponding modules to :meth:`__init__`


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
        report_progress: bool = False,
        progress_bar: bool = False,
        persist: bool = False,
        path: str = "",
        object_path: str = "/",
        neuron_eqs: List[str] = [],
        neuron_params: List[str] = [],
        neuron_eq_module: str = "differential_equations.eif_equations",
        neuron_param_module: string = "differential_equations.eif_parameters",
    ):
        """
        Initialize parameters used in setup and teardown.
        All data monitored by NeuronPopulation instances is automatically persisted as well as time steps of defaultclock.
        To add additional data set it on dictionary-like :attr:`BrianExperiment.persist_data`.

        :param dt: time step of default clock (:data:`brian2.core.clocks.defaultclock`) used as df for all :module:`brian2` objects
        :param report_progress:  print updates on respective stage of the experiment:
                                network definition, device collection, simulation, persisting
        :param progress_bar: show progress bar during simulation (cli) stage of experiment (see also param report progress)
        :param persist: specifies whether data is tb persisted
        :param path: path to the file to be used for persisting - if not set and persist=True will automatically generate file name 'experiments/experiment_i'
        :param object_path: path within the hdf5 file starting with root '/', eg. '/run_x/data'
        :param neuron_eqs: neuron equations to persist - verified against parameter neuron_eq_module
        :param neuron_params: neuron parameters to persist - verified against parameter neuron_param_module
        :param neuron_eq_module: module containing equations for neuron models - ensure that the equations used for the neuron and synapse models are contained therein
        :param neuron_param_module: module containing parameters for neuron models - ensure that the parameters used for the neuron and synapse models are contained therein
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
                module = importlib.import_module(self.neuron_eq_module)
                missing_names = [
                    eq for eq in neuron_eqs if eq not in module.__dict__.keys()
                ]
                if len(missing_names) > 0:
                    raise ValueError(
                        f"Some equations do not exist within {self.neuron_eq_module}: {missing_names}"
                    )
            if neuron_params:
                module = importlib.import_module(self.neuron_param_module)
                missing_names = [
                    eq for eq in neuron_params if eq not in module.__dict__.keys()
                ]
                if len(missing_names) > 0:
                    raise ValueError(
                        f"Some equations do not exist within {self.neuron_param_module}: {missing_names}"
                    )

        if (
            path != "" or object_path != "/" or neuron_eqs != [] or neuron_params != []
        ) and not persist:
            raise ValueError(
                "A dictionary for param persist_data or/and a path for persistence or/and an object_path or/adn neuron_eqs tb persisted"
                + " or/and neuron_params tb persisted were passed but persist is not set. Please set persist=True to persist data."
            )

        if os.path.isfile(path):
            raise ValueError(
                f"File {path} already exists. If you want to 'overwrite', please delete it manually."
            )

        self._persist = persist
        self._dt = dt

        self._path = os.path.abspath(path) if path else None
        self._opath = object_path
        self._neuron_equations = neuron_eqs
        self._neuron_parameters = neuron_params
        self._network = None

        self._timing = TimeTracker(verbose=report_progress)
        self._progress_bar = progress_bar

        self._meta = {}
        dt, unit_dt = convert_and_clean_brian2_quantity(self.dt)
        self._meta["dt"] = {"value": np.array(dt), "unit": unit_dt}

        # whether user is within context
        self._in_context = False

        self._device_context = []

        self._persist_data = BrianExperiment.PersistData(persist, self)

        self.neuron_eq_module = neuron_eq_module
        self.neuron_param_module = neuron_param_module

    @property
    def time_elapsed(self):
        """
        str representing time elapsed during simulation, None if :meth:`BrianExperiment.run()` not executed yet
        """
        return str({k: format_duration_ns(v) for k, v in self._timing.timings.items()})

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

        return retrieve_callers_frame(
            lambda fi: not (
                fi.function in dir(self.__class__)
                and os.path.abspath(fi.filename) == os.path.abspath(__file__)
            )
        )

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
            if issubclass(v.__class__, SpikeDeviceGroup)
            or issubclass(v.__class__, Synapse)
        ]:
            # get all StateMonitors, NeuronGroups, Synapses etc. (add Poisson)
            devices = [
                v
                for v in obj.__dict__.values()
                if (
                    v.__class__ == NeuronGroup
                    or v.__class__ == PoissonGroup
                    or v.__class__ == Synapses
                    or v.__class__ == StateMonitor
                    or v.__class__ == SpikeMonitor
                    or v.__class__ == PopulationRateMonitor
                )
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

        neuron_parameters_mod = importlib.import_module(self.neuron_param_module)
        namespace = clean_ns(neuron_parameters_mod.__dict__)
        namespace.update(clean_ns(self._retrieve_callers_context()))
        return namespace

    def run(self, t: Quantity = 0.01 * ms, report_freq=100 * ms):
        """
        run brian2 network via :meth:`brian2.network.run()`

        :param t: time for which the simulation is tb run
        :param report_freq: frequency at which the report is updated,
                            irrelevant if progress_report=False in :meth:`BrianExperiment.__init__()`
        """

        if not self._in_context:
            raise Exception(
                "Call only from within the context: with {self.__class__.__name__}() as exp:\n     exp.run()"
            )

        self._timing.add_timing(process="device_collection")

        if self._network == None:
            self._network = Network()
            for device in self._collect_devices():
                self._network.add(device)
        else:
            devices = set(self._collect_devices())
            for dev in devices.difference(self._network.objects):
                self._network.add(dev)

        self._timing.add_timing(process="simulation")

        if self._progress_bar:
            with TqdmCallBack(miniters=1, report_freq=report_freq) as tqdm:

                self._network.run(
                    t,
                    namespace=self._get_namespace(),
                    report=tqdm.update_cb,
                    report_period=report_freq,
                )
        else:
            self._network.run(t, namespace=self._get_namespace())

        self._timing.finalize()

        tt, unit = convert_and_clean_brian2_quantity(t)
        if "t" in self._meta.keys():
            tt += self._meta["t"]["value"]
        self._meta["t"] = {"value": np.array(tt), "unit": unit}

    def __enter__(self):

        # save local context to restore on exit and clear local context added within the context
        self._save_context()

        defaultclock.dt = self._dt

        self._in_context = True

        self._timing.add_timing("define_network")

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

            self._timing.add_timing("persistence")

            flmp = FileMap(self._path, mode="write", object_path=self._opath)
            with flmp as fm:

                # persist all data in the persist_data dictionary - will be overwritten if keys match automatically persisted entries
                if self.persist_data:
                    fm["persist_data"] = self.persist_data

                # persist all Neuronpopulations

                fm[SpikeDeviceGroup.__name__] = Node()
                neurp = fm[SpikeDeviceGroup.__name__]

                spike_devs = [
                    (k, v)
                    for k, v in self._retrieve_callers_context().items()
                    if isinstance(v, SpikeDeviceGroup)
                ]

                for i, (k, v) in enumerate(spike_devs):
                    neurp[k] = Node()
                    # neurp[k] = v.monitored

                    neurp[k]["ids"] = np.asarray(v.ids)

                    mon_data = v.monitored

                    for mon in mon_data.keys():
                        neurp[k][mon] = Node()
                        for var, val in mon_data[mon].items():
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
                    sn[k]["ids"] = np.asarray(v.synapses)
                    sn[k]["synapse_params"] = v.synapse_params

                    mon_data = v.monitored
                    for mon in mon_data.keys():

                        sn[k][mon] = Node()
                        for var, val in mon_data[mon].items():
                            sn[k][mon][var] = np.asarray(val)

                # persist all equations passed in __init__()
                if self._neuron_equations:
                    module = importlib.import_module(self.neuron_eq_module)
                    mod_name = BrianExperiment.resolve_module_name(module)
                    fm[mod_name] = Node()
                    for eq in self._neuron_equations:
                        fm[mod_name][eq] = module.__dict__[eq]

                # persist all parameters passed in __init__()
                if self._neuron_parameters:
                    module = importlib.import_module(self.neuron_param_module)
                    mod_name = BrianExperiment.resolve_module_name(module)
                    fm[mod_name] = Node()
                    for param in self._neuron_parameters:
                        fm[mod_name][param] = module.__dict__[param]

                # persist metadata
                fm["meta"] = self._meta

                # persist timing
                if self._timing:
                    fm["meta"]["run_times_in_ns"] = {
                        k: np.array(v) for k, v in self._timing.timings.items()
                    }
                frame_info = retrieve_callers_frame(
                    lambda fi: os.path.abspath(fi.filename) != os.path.abspath(__file__)
                )

                # save all parameters as we need them to resolve variables in equations
                module = importlib.import_module(self.neuron_param_module)

                fm["meta"]["neuron_parameters"] = {
                    k: dict(
                        zip(
                            ("value", "unit"),
                            convert_and_clean_brian2_quantity(module.__dict__[k]),
                        )
                    )
                    if isinstance(module.__dict__[k], Quantity)
                    else module.__dict__[k]
                    for k in module.__dict__.keys()
                    if not k in brian2.__dict__.keys()
                }

        self._timing.add_timing("reset_context")

        self._reset_context()

        self._timing.finalize()

        self._in_context = False

        if exc_type != None:
            raise exc_type(exc_value, traceback)
