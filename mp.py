import itertools
import multiprocessing
import multiprocessing.connection
from multiprocessing.sharedctypes import Value
import os
import re
import io
import time
import datetime
import signal
import sys
from typing import Callable, List, Dict, Any, Tuple, Union
import utils
import numpy as np
from functools import reduce

log_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), ".mp_log")


class Environment:
    """
    Implements Context Manager Interface in a functional style
    """

    def __init__(
        self, enter: Callable = None, on_error: Callable = None, exit: Callable = None
    ):
        """
        :param enter: function executed on entering the context
        :param on_error: function executed if an exception is raised
        :param exit: function executed on exit (whether an error is raised or not)
        """
        self.enter = enter
        self.on_error = on_error
        self.exit = exit

    def __enter__(self):
        if self.enter != None:
            self.enter()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type != None:
            if self.on_error != None:
                self.on_error()
        if self.exit != None:
            self.exit()
        if exc_type != None:
            raise exc_type(exc_value, traceback)


class Logger:
    """
    Implements Context Manager Interface for a logger
    """

    def __init__(self, path: str, stream_names: List[str], levels: List[str] = None):
        """
        :param path: path to which messages are logged
        :param levels: logging levels
        """
        err = utils.validate_file_path(path)
        if len(err) > 0:
            raise ValueError(f"{err}")
        self.path = path
        self.file = None
        self.stream_names = stream_names
        self.stream_max_digits = max([len(sn) for sn in stream_names])
        self.levels = levels
        self.level_max_digits = (
            max([len(l) for l in levels]) if self.levels != None else 0
        )
        self.proc_id_digits = int(np.ceil(np.log10(os.cpu_count() + 1)))

    def logall(
        self,
        process: int,
        stream_name: str,
        msgs: List[Tuple[str, float]],
        level: str = None,
    ):
        """
        :param process: process id
        :param stream_name: name of the stream
        :param msgs: messages with associated timestamps (POSIX timestamp from eg. :class:`time.time()`)
        :param level: logging level
        """
        for msg, tstmp in msgs:
            self.log(process, stream_name, tstmp, msg, level)

    def log(
        self, process: int, stream_name: str, tstamp: float, msg: str, level: str = None
    ):
        """
        :param process: process id
        :param stream_name: name of the stream
        :param tstamp: tstamp of the event (POSIX timestamp from eg. :class:`time.time()`)
        :param msg: msg
        :param level: logging level
        """
        if stream_name not in self.stream_names:
            raise ValueError(
                f"No such stream {stream_name} registered."
                + " Choose from {self.stream_names} passed to __init__()."
            )
        lvl = ""
        if level != None:
            if level in self.levels:
                lvl = level
            else:
                raise ValueError(
                    f"No such level {level}. Choose from {self.levels} passed to __init__()."
                )
        stream_name = (stream_name + " " * self.stream_max_digits)[
            : self.stream_max_digits
        ]
        lvl = (lvl + " " * self.level_max_digits)[: self.level_max_digits]
        self.file.write(
            f"[{datetime.datetime.fromtimestamp(tstamp).ctime()}] p {process:{self.proc_id_digits}.0f}  "
            + f"{lvl}  <{stream_name}>:    {msg}\n"
        )

    def __enter__(self):
        self.file = open(self.path, "w")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
        if exc_type != None:
            raise exc_type(exc_value, traceback)


class CaptureStandardStreams:
    """
    Implements Context Manager Interface for capturing standard streams stdout and stderr
    """

    def __init__(
        self,
        stdout: Union[bool, str] = True,
        stderr: Union[bool, str] = False,
    ):
        self.stderr = None
        self.stdout = None
        self._err_stream = None
        self._out_stream = None
        self.df_stdout = None
        self.df_stderr = None

        self._init_out_path = None
        self._init_err_path = None

        if isinstance(stdout, bool):
            self.capture_out = stdout
        else:
            self.capture_out = True
            stdout = os.path.abspath(stdout)
            if not os.path.isdir(os.path.dirname(stdout)):
                raise ValueError(
                    f"Directory {os.path.dirname(stdout)} in path does not exist."
                )
            self._init_out_path = stdout

        if isinstance(stderr, bool):
            self.capture_err = stdout
        else:
            self.capture_err = True
            stderr = os.path.abspath(stderr)
            if not os.path.isdir(os.path.dirname(stderr)):
                raise ValueError(
                    f"Directory {os.path.dirname(stderr)} in path does not exist."
                )
            self._init_err_path = stderr

    def __enter__(self):

        if self.capture_out:
            self._out_stream = (
                io.StringIO()
                if self._init_out_path == None
                else open(self._init_out_path, mode="w+")
            )
            self.df_stdout = sys.stdout
            sys.stdout = self._out_stream

        if self.capture_err:
            self._err_stream = (
                io.StringIO()
                if self._init_err_path == None
                else open(self._init_err_path, mode="w+")
            )
            self.df_stderr = sys.stderr
            sys.stderr = self._err_stream

    def __exit__(self, exc_type, exc_value, traceback):

        if self.capture_out:
            sys.stdout.flush()
            sys.stdout = self.df_stdout

            self._out_stream.seek(0)
            self.stdout = self._out_stream.read()
            self._out_stream.close()

        if self.capture_err:
            sys.stderr.flush()
            sys.stderr = self.df_stderr

            self._err_stream.seek(0)
            self.stderr = self._err_stream.read()
            self._err_stream.close()

        if exc_type != None:
            raise exc_type(exc_value, traceback)


class MultiPipeCommunication:
    """
    create two linked communication objects with a set of pipes available as attributes under the provided stream names
    eg. for a stream name 'name' parent_com_obj.name and child_com_obj.name will return either end of the associated pipe
    """

    class Communication:
        def __init__(
            self,
            control_pipe: multiprocessing.connection.Connection,
            streams: Dict[str, multiprocessing.connection.Connection],
        ):

            self.streams = streams
            self.streams["control"] = control_pipe

            for p_name in streams.keys():
                setattr(self, p_name, self.streams[p_name])

            self.control_signals = {"TERMINATE": False}
            self.END_OF_STREAM = "END_OF_STREAM"

        def send(self, msg: str, stream: multiprocessing.connection.Connection):
            stream.send((msg, time.time()))

        def poll(self, stream: multiprocessing.connection.Connection, wait: int = 0):
            return stream.poll(wait)

        def recv(
            self, stream: multiprocessing.connection.Connection
        ) -> List[Tuple[str, float]]:
            """
            :param stream: stream tb received from
            :return: messages with associated timestamps
            """
            msgs = []
            while self.poll(stream=stream):
                msgs.append(stream.recv())
            return msgs

        def recvlines(
            self, stream: multiprocessing.connection.Connection, keep_empty: bool = True
        ) -> List[Tuple[List[str], float]]:
            """
            Receive by line

            :param stream: stream tb received from
            :param keep_empty: whether or not to keep empty lines
            :return: messages line by line with associated timestamps
            """
            # remove final '\n' if exists
            msgs = [
                (m[:-1], t) if m.endswith("\n") else (m, t)
                for m, t in self.recv(stream)
            ]

            return [
                ([m for m in mg.split("\n") if keep_empty or len(m) > 0], t)
                for mg, t in msgs
            ]

        def closed(self, stream: multiprocessing.connection.Connection):
            return stream.closed()

        def close(self):
            for stream in self.streams.values():
                stream.close()

        def send_terminate(self):
            self.control.send("TERMINATE")

        def should_terminate(self):
            self.recv_control()
            return self.control_signals["TERMINATE"]

        def recv_control(self):
            if self.control.poll(0):
                msg = [
                    m.strip()
                    for m in self.control.recv().split("\n")
                    if len(m.strip()) > 0
                ]
                for m in msg:
                    if m not in self.control_signals.keys():
                        raise Exception("No such control signal known.")
                    self.control_signals[m] = True

    def __new__(cls, stream_names: List[str]):
        """
        :param str: list of stream names for which pipe ends are to be set on the communication objects,
            eg. for 'name' com_obj.name will return associated pipe end
        :return: communication object for parent process and communication object for child process with pipes set according to parameter streams
        """
        if len(stream_names) == 0:
            raise ValueError("Provide at least one stream.")

        p_pipe_control, c_pipe_control = multiprocessing.Pipe()

        p_streams = {}
        c_streams = {}

        for sn in stream_names:
            p_streams[sn], c_streams[sn] = multiprocessing.Pipe()

        parent_com = MultiPipeCommunication.Communication(p_pipe_control, p_streams)
        child_com = MultiPipeCommunication.Communication(c_pipe_control, c_streams)
        return parent_com, child_com


class ProcessExperiment(multiprocessing.Process):

    """
    Process executing a target function (parameter target) and communicating stdout, stderr and current file path being processed
    via dedicated pipes wrapped in an instance of :class:`MultiPipeCommunication`
    """

    supported_stream_names = ["stdout", "stderr", "curfile"]

    def __init__(
        self,
        idx: int,
        num_procs: int,
        target: Callable,
        kwargs: Dict[str, Any],
        com: MultiPipeCommunication = None,
    ):
        """
        :param idx: index of process in process group
        :param num_procs: total number of processes in process group
        :param target: target function executed for each element of the cartesian product of the value ranges in parameter parameters
                        whose signature is filled using parameter kwargs together with a specific instance of the cartesian
                       product of the value ranges in parameter parameters
                       and a file path generated from  such an instance and parameter base_path by parameter file_name_generator
                       - note the values of the specific instance and the generated file path take precedence over parameter kwargs
                       - signature: parameters are passed as key-word arguments only - additional kwarg with name 'path' is passed
                            - therefore it is crucial to use the same exact names for parameters of the target function:
                            names in parameter kwargs, names in parameter parameters, 'path'
                            (note that the kwargs are updated in that order)
        :param kwargs: keyword arguments used to fill signature of target function
        :param com: communication object for communication with parent process may provide a subset of :attr:`ProcessExperiment.supported_stream_names`
                            streams
        """

        self.idx = idx
        self.num_procs = num_procs

        self.target = target
        self.kwargs = kwargs

        self.com = com

        super().__init__()

    def close(self):
        self.com.close()
        super().close()

    def run(self):
        """
        execute target function (parameter target) for each nth instance with offset p of the cartesian product of the value ranges in parameter parameters,
        where p is the index of the process and n is the number of processors available
        """
        if hasattr(self.com, "curfile"):
            self.com.send(self.kwargs["path"], stream=self.com.curfile)

        capture_streams = CaptureStandardStreams(stdout=True, stderr=True)
        with capture_streams:
            self.target(**self.kwargs)

        if hasattr(self.com, "stdout"):
            self.com.send(capture_streams.stdout, stream=self.com.stdout)
        if hasattr(self.com, "stderr"):
            self.com.send(capture_streams.stderr, stream=self.com.stderr)
        if hasattr(self.com, "curfile"):
            self.com.send(self.com.END_OF_STREAM, stream=self.com.curfile)


def float_to_path_component(x: float):
    return f"{x:.2f}".replace(".", "-")


def path_component_to_float(pc: str):
    return float(pc.replace("-", "."))


def file_name_generator(instance):
    """
    Generate a file name from key value pairs
    (reversed by :func:`file_name_parser`)
    """
    return (
        "_".join(
            [
                f"{c}_{float_to_path_component(v)}"
                if isinstance(v, float)
                else f"{c}_{v}"
                for c, v in instance
            ]
        )
        + ".h5"
    )


def file_name_parser(fname: str):
    """
    Parse key value pairs from a file name
    (reversed by :func:`file_name_generator`)
    """
    pattern = "([^_]+)_([0-9]+-[0-9]+)|([^_]+)_([0-9]+)|([^_]+)_([^_-]+)"

    pattern_verif = f"(({pattern})_)*({pattern})\\.h5"

    if re.fullmatch(pattern_verif, fname) == None:
        raise ValueError(
            f"file name {fname} does not conform to pattern {pattern_verif}."
        )

    base = fname[: -len(".h5")]

    m = re.compile(pattern)
    variables = m.findall(base)

    parameters = []
    for flname, flval, iname, ival, sname, sval in variables:
        if len(flname) > 0:
            parameters.append((flname, path_component_to_float(flval)))
        elif len(iname) > 0:
            parameters.append((iname, int(ival)))
        elif len(sname) > 0:
            parameters.append((sname, sval))

    return parameters


class Progress:
    def __init__(self, n: int):
        self.n = n
        self.len_format_n = 1
        if n > 1:
            self.len_format_n = (
                int(np.ceil(np.log10(n)))
                if n % 10 != 0
                else int(np.ceil(np.log10(n))) + 1
            )
        self.t = time.time_ns()
        self.schema_printed = False

    def update(self, it):
        def pad_str(s: str, n: int):
            # return padded string and negative offset (if length of string > n)
            return ((s + " " * n)[0:n], 0) if n >= len(s) else (s, len(s) - n)

        spcg = "      "

        values, labels, unit_lengths = utils.split_into_temporal_components(
            time.time_ns() - self.t, full=True
        )
        values, labels, unit_lengths = values[3:7], labels[3:7], unit_lengths[3:7]
        t_str = Progress.format_duration(values, labels, unit_lengths)

        if not self.schema_printed:
            vals = [f"{v:{ul}d}" for v, ul in zip(values, unit_lengths)]
            labels = [
                (" " * len(f"{v}") + l)[len(l) :] if len(f"{v}") > len(l) else l
                for l, v in zip(labels, vals)
            ]
            str_comps = []
            lbls = [f"[{'-'.join(labels[::-1])}]", "[%]", "it"]
            vals = [
                f"[{t_str[1]}]",
                f"{it/self.n:2.2f}",
                f"{it:{self.len_format_n}d}",
            ]
            for i, (k, v) in enumerate(
                zip(
                    lbls,
                    vals,
                )
            ):
                c, noff = pad_str(k, len(v))
                if i == len(lbls) - 1:
                    str_comps.append(f"{c}")
                else:
                    str_comps.append(
                        f"{c}{spcg[:-noff if noff > 0 else len(spcg)] if noff <= len(spcg) else ''}"
                    )
            print("".join(str_comps) + " / total\n")
            self.schema_printed = True

        print(
            f"[{t_str[1]}]{spcg}{it/self.n * 100.0:2.2f}{spcg}{it:{self.len_format_n}d} / {self.n}",
            end="\r" if it != self.n else "\n",
        )

    @staticmethod
    def format_duration(values, labels, unit_lengths):
        """
        create string representation of time duration given the time increments (labels), their respective values (values) and their unit lengths (unit_lengths)

        :return: string representation of time with format: y, d, h, m, s - where only duration components whose first increment is reached are used
        """
        comps = list(zip(values, labels, unit_lengths))[::-1]
        return "-".join([f"{l}" for _, l, _ in comps]), "-".join(
            [f"{c:{ul}d}" for c, _, ul in comps]
        )


class Pool:
    """
    Pool of instances of :class:`ProcessExperiment`
    """

    def __init__(
        self,
        base_path: str,
        parameters: Dict[str, List[Any]],
        target: Callable,
        kwargs: Dict[str, Any],
        log_path: str = log_path,
        num_procs: int = None,
        file_name_generator: Callable[
            [List[Tuple[str, Any]]], str
        ] = file_name_generator,
        progress: bool = True,
    ):
        """
        distribute the value ranges equally across as many processes as there are available cpus and execute the target function
        for each instance of the cartesian product of the process-specific subset of value ranges in each process in parallel

        :param base_path: base directory which is used in generating a file path by the target function (parameter target)
        :param parameters: mapping of parameter names to value ranges which are distributed equally across as many processes
                        as there are cpus available and for which the target function (parameter target) is called for each
                            instance of the cartesian product of the ranges in the respective process
        :param target: target function executed for each element of the cartesian product of the value ranges in parameter parameters
                        whose signature is filled using parameter kwargs together with a specific instance of the cartesian
                        product of the value ranges in parameter parameters
                        and a file path generated from  such an instance and parameter base_path by parameter file_name_generator
                        - note the values of the specific instance and the generated file path take precedence over parameter kwargs
                        - signature: parameters are passed as key-word arguments only - additional kwarg with name 'path' is passed
                                - therefore it is crucial to use the same exact names for parameters of the target function:
                                names in parameter kwargs, names in parameter parameters, 'path'
                                (note that the kwargs are updated in that order)
        :param kwargs: keyword arguments used to fill signature of target function
        :param log_path: path to logfile for subprocesses
        :param num_procs: number of processes tb used - for high load num_procs ~ processor count - df: num_procs == processor count
        :param file_name_generator: function generating a file name (not a path) for an instance
                                    of the cartesian product of the value ranges in parameter parameters
                                    - default: :attr:`ProcessExperiment.log_path` is used
        :param progress: show a continuously updating progress bar
        """

        base_path = os.path.abspath(base_path)
        if not os.path.isdir(base_path):
            raise ValueError(f"No such directory: {base_path}.")

        if len(parameters.keys()) == 0 or not all(
            [len(v) > 0 for p, v in parameters.items()]
        ):
            raise ValueError(
                "parameters is either empty or length of one of the values is 0."
            )

        self.base_path = base_path
        self.target = target
        self.kwargs = kwargs
        self.log_path = log_path
        self.num_procs = num_procs if num_procs != None else os.cpu_count()
        self.file_name_generator = file_name_generator

        self.plabels, self.pvalues = zip(
            *[(p, parameters[p]) for p in parameters.keys()]
        )
        self.instances = itertools.product(*self.pvalues)

        for instance in itertools.product(*self.pvalues):
            instance = list(zip(self.plabels, instance))
            fn_params = []
            for k, v in list(self.kwargs.items()) + instance:
                if np.isscalar(v):
                    fn_params.append((k, v))
                elif isinstance(v, dict) and any(
                    [np.isscalar(vv) for vv in v.values()]
                ):
                    for kk, vv in v.items():
                        if np.isscalar(vv):
                            fn_params.append((kk, vv))
            fname = self.file_name_generator(fn_params)
            fpath = os.path.join(base_path, fname)
            err = utils.validate_file_path(fpath, ext=".h5")
            if len(err) > 0:
                raise ValueError(
                    f"file_name_generator generated an invalid file_path {fpath} given {instance} producing the error '{err}'."
                )

        self.num_params = reduce(
            lambda acc, x: acc * x, [len(vl) for vl in self.pvalues], 1
        )
        self.progress = progress

    def next_process(self, idx):
        exhausted = False
        try:
            instance = next(self.instances)
        except StopIteration:
            exhausted = True

        proc = None
        p_com = None

        if not exhausted:
            instance = list(zip(self.plabels, instance))
            p_com, c_com = MultiPipeCommunication(
                stream_names=ProcessExperiment.supported_stream_names
            )
            kwg = self.kwargs.copy()

            for p, v in instance:
                kwg[p] = v

            fn_params = []
            for k, v in list(self.kwargs.items()) + instance:
                if np.isscalar(v):
                    fn_params.append((k, v))
                elif isinstance(v, dict) and any(
                    [np.isscalar(vv) for vv in v.values()]
                ):
                    for kk, vv in v.items():
                        if np.isscalar(vv):
                            fn_params.append((kk, vv))
            # [(k,v) for k,v in (list(self.kwargs.items()) + instance).items() if np.isscalar(v)]
            fname = self.file_name_generator(fn_params)

            fpath = os.path.join(self.base_path, fname)

            kwg["path"] = fpath

            proc = ProcessExperiment(
                idx,
                self.num_procs,
                self.target,
                kwg,
                c_com,
            )
        return exhausted, (proc, p_com)

    def update_signal_handler(self, processes):
        signal.signal(signal.SIGINT, Pool.create_signal_handler(processes))

    def run(self):

        processes = []
        exhausted = False

        for idx in range(self.num_procs):
            exhausted, p = self.next_process(idx)
            if not exhausted:
                processes.append(p)
            else:
                break

        self.update_signal_handler(processes)

        stdouts = {p.idx: [] for p, _ in processes}
        stderrs = {p.idx: [] for p, _ in processes}

        # reverse order of list processes to keep processing order of processes consistent, ie lower indices first,
        #   in main loop below, as we delete from the list we are iterating over (in main loop below),
        #   we need to process the list elements in reverse order - effectively reversing order twice
        processes = processes[::-1]

        for p, _ in processes[::-1]:
            p.start()

        progress = None
        it = 0
        if self.progress:
            progress = Progress(self.num_params)
        progress.update(it)

        with Logger(path=log_path, stream_names=["stdout", "stderr"]) as logger:

            while len(processes) > 0:

                for i in range(len(processes) - 1, -1, -1):

                    p, com = processes[i]

                    if com.poll(stream=com.stdout, wait=1 / len(processes)):
                        msg = com.recvlines(stream=com.stdout)
                        stdouts[p.idx] += msg
                        logger.logall(p.idx, "stdout", msg)

                        if self.progress:
                            it += 1
                            progress.update(it)

                        if com.poll(stream=com.stderr, wait=0.05):
                            msg = com.recvlines(stream=com.stderr)
                            stderrs[p.idx] += msg
                            logger.logall(p.idx, "stderr", msg)

                        # start next task
                        p.join()
                        p.close()
                        com.close()
                        idx = p.idx
                        exhausted, (p, com) = self.next_process(idx)
                        if exhausted:
                            del processes[i]
                            self.update_signal_handler(processes)
                        else:
                            processes[i] = (p, com)
                            self.update_signal_handler(processes)
                            p.start()

                    if progress:
                        progress.update(it)

        return stdouts, stderrs

    @staticmethod
    def create_signal_handler(processes: List[ProcessExperiment]):
        parent_pid = multiprocessing.current_process().pid

        def signal_handler(signal, frame):
            if multiprocessing.current_process().pid == parent_pid:
                print("\nExiting parent gracefully ...\n")
                print("\nterminating child processes (brute force)...\n")

                for p, com in processes:
                    # ideally a more graceful exit chosen eg using multiprocessing.Event
                    #  however brian2 does not allow graceful exiting
                    if p.is_alive():
                        if com.poll(stream=com.curfile):
                            msgs = com.recvlines(stream=com.curfile)
                            if len(msgs) > 0 and msgs[-1][0] != com.END_OF_STREAM:
                                fpath = msgs[-1][0]
                                if os.path.isfile(fpath):
                                    os.remove(fpath)
                        p.terminate()

                for p, com in processes:
                    p.join()
                    p.close()
                    com.close()

                print("\nChild processes terminated.")
                print("\nExiting.")
                sys.exit(0)

            else:
                # child process
                print(
                    f"Signal handler: child w/ pid {multiprocessing.current_process().pid} doing nothing."
                )

        return signal_handler
