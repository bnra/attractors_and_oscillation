import os
import shutil
import re
from typing import Any, Callable, Dict, List, Tuple, Union
import inspect
import brian2
from brian2.units.fundamentalunits import Quantity, get_unit, Unit
import numpy as np
import subprocess


class TestEnv:
    """
    Implements Context Manager Interface to setup and teardown a test environment
    for file i/o
    """

    def __init__(self, path=".tmp"):
        """
        :param path: path to be used as temporary directory (default: '.tmp')
        """
        self.initial_dir = os.path.abspath(".")
        self.tmp_dir = os.path.abspath(path)

    def __enter__(self):
        """
        makes the tmp_dir and makes it the cwd
        """
        os.makedirs(self.tmp_dir)
        os.chdir(self.tmp_dir)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        makes the initial_dir the cwd and deletes the tmp_dir
        """
        os.chdir(self.initial_dir)
        shutil.rmtree(self.tmp_dir)


        if exc_type != None:
            raise exc_type(exc_value, traceback)


def run_cmd(cmd):
    p = subprocess.Popen(
        [cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    out, err = p.communicate()
    return out.decode("utf-8"), err.decode("utf-8")


def validate_file_path(path: str, ext: str = ""):
    """
    Validate file path -  whether
    base path exists,
    file name has correct extension [verified only in case ext passed],
    enforces naming conventions on basename only containing characters [a-zA-Z0-9_-]
    yet may start with '.' (hidden files)
    and has a maximal length of 255
    (https://www.ibm.com/docs/en/aix/7.1?topic=files-file-naming-conventions)


    :param path: path whose validity is tb verified
    :param ext:  file extension - validity of the extension not verified

    :return: error message - empty if path valid
    """
    base_path, head = os.path.split(os.path.abspath(path))

    if not os.path.isdir(base_path):
        return f"Basepath { base_path } of parameter path is not a valid directory."
    if not head.endswith(ext):
        return f"Filename { head } does not have the correct extension { ext }."
    fname = head
    if ext != "":
        fname = head[: -len(ext)]

    if len(fname) > (255 - len(ext)) or not re.fullmatch("\\.?[a-zA-Z0-9_-]+", fname):
        return f"Base file name { fname } must be of length <= 255 and contain only characters [a-zA-Z0-9_-] yet may start with '.' (hidden)."
    return ""


def generate_sequential_file_name(base_path: str, base_name: str, ext: str):
    """
    Generate the next file name sequentially given a directory name and base file name
    If directory does not exist the directory is created.

    :param base_path: directory where the files are tb created
    :param base_name: basis of the file name used in sequential generation, file name is base_name + '_' + current increment
    """
    i = 0
    while os.path.isfile(os.path.join(base_path, f"{base_name}_{i}{ext}")):
        i += 1
    return os.path.join(base_path, f"{base_name}_{i}{ext}")


def retrieve_callers_frame(condition: Callable[[inspect.FrameInfo], bool]):
    """
    retrieve the frame satisfying a condition - if no such frame raises Exception

    :param condition: condition to test for the frame in question
    :return: first frame in stack passing the condition
    """
    # find first caller in call stack (excepting top most frame ~ call to this fct)fullfilling condition of the parameter condition

    # top most stack frame represents call to this function

    for frame_info in inspect.stack()[1:]:
        # print(frame_info)
        if condition(frame_info):
            return frame_info
    raise Exception(f"No frame satisfying condition { condition } found.")


def retrieve_callers_context(frame_info: inspect.FrameInfo):
    """
    retrieve the context for a frame
    - context: globals updated with locals

    :param frame_info: the information object of a frame for which context is tb retrieved
    :return: context of the respective frame of the encapsulating information  object
    """
    # retrieve the context: globals updated with locals (ie locals shadow globals if same key in both)
    frame = frame_info.frame
    return {k: v for k, v in [*frame.f_globals.items()] + [*frame.f_locals.items()]}


def clean_brian2_quantity(x: Quantity) -> Tuple[np.ndarray, str]:
    """
    clean quantity of its unit

    :param x: quantity cleaned of its unit
    :return: cleaned quantity with unit scaling, and string representation of the unit
    """
    unit = x.get_best_unit()
    return x / unit, str(unit)


def convert_and_clean_brian2_quantity(x: Quantity) -> Tuple[np.ndarray, str]:
    """
    convert quantity to base unit and clean of its base unit

    :param x: quantity which is tb converted to base unit and then cleaned of its unit
    :return: cleaned quantity with base unit scaling, and string representation of the base unit
    """
    # copies (np.asarray(x) is in-place)
    unit = get_brian2_base_unit(x)
    x_base = np.asarray(x)

    # if it's a scalar value (eg. array(3) - scalar/not nested - and not array([3]) - single nested value) unwrap else return np.ndarray
    return (x_base.item(), str(unit)) if len(x_base.shape) == 0 else (x_base, str(unit))


def unwrap_brian2_variable_view(
    x: brian2.core.variables.VariableView,
) -> Union[np.ndarray, Quantity]:
    """
    unwrap instance of :class:`brian2.core.variables.VariableView`

    :param x: instance of :class:`brian2.core.variables.VariableView` tb unwrapped
    :return: variable value
    """
    return x[:]


def get_brian2_unit(x: Quantity) -> Unit:
    """
    get brian2 unit of quantity

    :param x: quantity for which unit is tb determined
    :return: unit of quantity parameter x
    """
    return x.get_best_unit()


def get_brian2_base_unit(x: Quantity) -> Unit:
    """
    get brian2 base unit of quantity - basic unscaled unit

    :param x: quantity for which base unit is tb determined
    :return: base unit of quantity parameter x
    """
    return get_unit(x.dim)


class Brian2UnitError(Exception):
    """
    when instance of :class:`Brian2.Unit` does not match the expected unit
    """

    pass


def split_into_temporal_components(t: int, full=False):
    """
    split into significant temporal components (significant up until the largest non-zero component)
    :param full: if set returns all temporal components
    :return: values and labels of temporal components [ns, mu_s, ms, s, m, h, d, y]
    """
    unit_length = {
        "ns": 3,
        "mu_s": 3,
        "ms": 3,
        "s": 2,
        "m": 2,
        "h": 2,
        "d": 3,
        "y": 4,
    }
    t, ns = divmod(t, 1000)
    t, mu_s = divmod(t, 1000)
    t, ms = divmod(t, 1000)
    t, s = divmod(t, 60)
    t, min = divmod(t, 60)
    t, h = divmod(t, 24)
    y, d = divmod(t, 365)
    labels = ["ns", "mu_s", "ms", "s", "m", "h", "d", "y"]
    values = [ns, mu_s, ms, s, min, h, d, y]
    final_labels = []
    found_nz = False
    if not full:
        for i in range(len(labels) - 1, -1, -1):
            if found_nz or values[i] != 0:
                final_labels.append(labels[i])
                found_nz = True
        final_labels = final_labels[::-1]
    else:
        final_labels = labels
    return (
        values[: len(final_labels)],
        final_labels,
        [unit_length[l] for l in final_labels],
    )


def format_duration_ns(t: int, drop_zeros=True):
    """
    create string representation of time duration in y, d, h, m, s, ms, mu_s, ns (y:years ~ 365 days)

    :param t: time elapsed in nanoseconds (eg. as a difference of time points or since epoch see :meth:`time.time_ns()`)
    :return: string representation of time with format: y, d, h, m, s, ms, mu_s, ns - where only duration components whose first increment is reached are used
    """
    (
        values,
        labels,
        unit_lengths,
    ) = split_into_temporal_components(t)
    comps = zip(values, labels, unit_lengths)
    if drop_zeros:
        comps = [(values[0], labels[0], unit_lengths[0])]
        for c, l, ul in zip(values[1:], labels[1:], unit_lengths[1:]):
            if c != 0:
                comps.append((c, l, ul))
            else:
                break
    return "  ".join([f"{c:{ul}d} {l}" for c, l, ul in comps[::-1]])


def unique_idx(x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    compute unique values and all indices for each unique value (efficient)

    :param x: 1D array-like object that holds all values
    :return: unique values and indices for each unique value, where the ith set of indices contains all indices of the ith value
    """
    if len(x.shape) > 1:
        raise ValueError("Only 1D np.ndarrays supported")
    indices = np.argsort(x)

    vals, idx = np.unique(x[indices], return_index=True)
    return vals, np.split(indices, idx)[1:]


def values_in_interval(t0: float, t1: float, dt: float):
    """
    compute the number of values in the interval [t0,t1)

    :param t0: start of interval (incl.)
    :param t1: end of interval (excl.)
    :param dt: step size of a step
    :return: number of values (= number of steps + 1) in interval from t1 to t0 given step size dt
    """
    return int(np.ceil((t1 - t0) / dt))


def next_power_of_two(x: int):
    """
    next power of two implemented using bit length of integer
    (equivalent to ceil(log2(x)), ie smallest sp such that x <= 2 ** sp)

    :param x: value for which the next largest power of 2 is tb determined
    :return: smallest power of two greater equal to x (smallest sp such that  x <= 2 ** sp)
    """
    # shifting x to the left by 1 for bits results in bit(e) = x = sp(e) (incl shift for bit(.)),
    #  ie bit'(x-1) = sp'(x) where bit', sp' are the respective inverse fct
    # bit -> x :             1 -> (0,1), 2 -> (2,3), 3 -> (4,5,6,7), 4 -> (8, ..., 15)
    # sp  -> x : 0 -> (0,1), 1 -> (2),   2 -> (3,4), 3 -> (5,6,7,8)
    return int(x - 1).bit_length()


def round_to_res(x: float, res: float):
    """
    round to a resolution of the most significant digit of parameter res
    (rounded to the number of decimals at which res has the first nz value, eg. 0.001 -> 3 decimals)

    :param x: number tb rounded
    :param res: most significant bit of this resolution specifies the resolution of rounding
    :return: number rounded to a resolution of the most significant digit of param res
    """
    if res <= 0.0 or res > 1.0:
        raise ValueError(f"Param res must be in (0,1]. Is {res}")
    return round(x, -int(np.log10(abs(res))))


def compute_time_interval(
    t: float, dt: float, t_start: float = None, t_end: float = None
):
    """
    compute a time interval [t_start, t_end] (closed bounds)
    given the optional specifications (t_start, t_end) and verify its validity


    :param t: simulation time [ms]
    :param dt: simulation time step [ms]
    :param t_start: time lower bound
    :param t_end: time upper bound
    :return: bounds of the interval, last time point
    """
    last_idx = values_in_interval(0.0, t, dt) - 1
    t_last = round_to_res(last_idx * dt, dt)
    t_end = t_end if t_end != None and t_end <= t_last else t_last
    t_start = t_start if t_start != None else 0.0

    if t_start >= t_end:
        raise ValueError(
            f"t_start cannot be >= t_end. Are: t_start {t_start} and t_end {t_end}."
        )
    return t_start, t_end, t_last


def restrict_to_interval(
    x: np.ndarray, dt: float, t_start: float = None, t_end: float = None
) -> Tuple[np.ndarray, float, float]:
    """
    restrict data to interval [t_start, t_end] given data is sampled at equidistant intervals of dt

    :param x: data tb restricted to interval
    :param dt: time step at which data is sampled
    :param t_start: incl. lower bound (time [ms]) for the restriction interval
    :param t_end: incl. upper bound (time [ms]) for the restriction interval
                    - if t_end > simulation time ~ t_end=None
    :return: data in interval [t_start, t_end], t_start, t_end
    """
    if t_end == None:
        idx_end = x.shape[0] - 1
    else:
        idx_end = round(t_end / dt)
        if idx_end >= x.shape[0]:
            idx_end = x.shape[0] - 1

    if t_start == None:
        idx_start = 0
    else:
        idx_start = int(t_start / dt)

    return (
        x[idx_start : idx_end + 1],
        round_to_res(idx_start * dt, dt),
        round_to_res(idx_end * dt, dt),
    )


def logical_and(*args):
    if len(args) < 2:
        raise ValueError("Pass at least two boolean np.ndarrays.")
    idx = args[0]
    for a in args[1:]:
        idx = np.logical_and(idx, a)
    return idx
