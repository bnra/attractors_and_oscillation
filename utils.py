import os
import shutil
import re
from typing import Callable
import inspect
from brian2.units.fundamentalunits import Quantity


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

    def __exit__(self, exc_type, exc_value, traceback):
        """
        makes the initial_dir the cwd and deletes the tmp_dir
        """
        os.chdir(self.initial_dir)
        shutil.rmtree(self.tmp_dir)

        # proc = psutil.Process()
        # fds = proc.open_files()
        # for f in fds:
        #    os.close(f.fd)

        if exc_type != None:
            raise exc_type(exc_value, traceback)


def validate_file_path(path: str, ext: str = ""):
    """
    Validate file path -  whether
    base path exists,
    file name has correct extension [verified only in case ext passed],
    enforces naming conventions on basename only containing characters [a-zA-Z0-9_], length of 255
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
    if ext:
        fname = head[: -len(ext)]

    if len(fname) > (255 - len(ext)) or not re.fullmatch("[a-zA-Z0-9_]+", fname):
        return f"Base file name { fname } must be of length <= 255 and contain only characters [a-zA-Z0-9_]."
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


def retrieve_callers_frame(condition:Callable[[inspect.FrameInfo],bool]):
    # find first caller in call stack (excepting top most frame ~ call to this fct)fullfilling condition of the parameter condition

    # top most stack frame represents call to this function
    for frame_info in inspect.stack()[1:]:
        if condition(frame_info):
            return frame_info
    raise Exception(
        f"No frame satisfying condition { condition } found."
    )

def retrieve_callers_context(frame_info:inspect.FrameInfo):
    # retrieve the context: globals updated with locals (ie locals shadow globals if same key in both)
    frame = frame_info.frame
    return {k: v for k, v in [*frame.f_globals.items()] + [*frame.f_locals.items()]}

def clean_brian2_quantity(x:Quantity):
    unit = x.get_best_unit()
    return x / unit, str(unit)

