"""
Entrypoint for file persistence with h5 files

Additionally exports:
    - persistence.opath - :mod:`opath` provides utilities for dealing with object paths
    - persistence.validate_file_path  - :func:`utils.validate_file_path` validates file paths
    - persistence.generate_sequential_file_name - :func:`utils.generate_sequential_file_name` generates file name sequentially
"""

import numpy as np
import tables
import warnings

import re
import sys
import os
import json
from typing import Iterable, Tuple, List, Union, Dict, Any

from utils import validate_file_path, generate_sequential_file_name
import opath

# enable from persistence import x
sys.modules["persistence.opath"] = opath
sys.modules["persistence.validate_file_path"] = validate_file_path
sys.modules["persistence.generate_sequential_file_name"] = generate_sequential_file_name


class Array:
    """
    Placeholder for :class:`tables.array.Array()`
    used by Reader and Writer class to enable Mapping Interface while also allowing arbitrary
    nesting

    :param obj:  array to be stored
    """

    def __init__(self, obj: np.ndarray, *args, **kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs


class VArray:
    """
    Placeholder for :class:`tables.vlarray.VlArray()`
    used by Reader and Writer class to enable Mapping Interface while also allowing arbitrary
    nesting

    :param obj:  array to be stored (optional)
    """

    def __init__(self, *args, obj: Union[np.ndarray, None] = None, **kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs


class Node:
    """
    Placeholder for :class:`tables.groups.Group()`
    used by Reader and Writer class to enable Mapping Interface while also allowing arbitrary nesting
    """

    def __init__(self):
        pass


class Reader:
    """
    Implements a Mapping Interface for the passed h5 file enabling indexing by key
    """

    def __init__(self, file: tables.file.File, object_path: str):
        """
        :param object_path: object path in object tree in hdf5 file (path must already exist in file)
        :param file: underlying :class:`tables.file.File` which interfaces with the hdf5 file
        """

        self.opath = object_path
        self.file = file

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        for node in self.keys():
            yield node

    def up(self):
        components = opath.split(self.opath)
        if len(components) == 1:
            raise opath.OpathError(f"Cannot move up on Node root '/': { self.opath }")
        path = opath.join(*components[:-1]) if len(components) > 2 else components[0]
        return self.__class__(self.file, path)

    def _extract_value(
        self,
        key: str,
        nodes: Tuple[Dict[str, tables.group.Group]],
        leaves: Dict[str, Union[tables.array.Array, tables.vlarray.VLArray]],
        recursive: bool = False,
    ):
        """
        extract values

        :param recursive: whether to read all descendants into memory recursively
        :return: instance of :class:`persistence.Reader` or a terminal node read into memory,
                 if recursive set will return dictionary with all descendants read into memory or
                 a terminal node read into memory
        """
        if key in nodes.keys():
            if recursive:
                return self.__class__(self.file, opath.join(self.opath, key)).load()

            return self.__class__(self.file, opath.join(self.opath, key))
        else:
            if key in leaves.keys():
                x = leaves[key].read()
                # convert byte strings to unicode
                if not isinstance(x, np.ndarray):
                    raise ValueError(
                        f"{self.__class__.__name__} can only read .h5 files with leaf nodes of type numpy.ndarray. Is {type(x)}"
                    )
                # determining byte strings in numpy: '[=<>|]S[0-9]+' where [0-9]+ is length of longest string, [=<>|] byte order
                if re.fullmatch("[=<>|]S[0-9]+", x.dtype.str):
                    x = x.astype(dtype=str)
                    # unpack single strings
                    if x.size == 1:
                        x = x[0]
                return (
                    x
                    if not re.fullmatch("[=<>|]S[0-9]+", x.dtype.str)
                    else x.astype(dtype=str)
                )
            else:
                raise KeyError(f"Key { key } not contained in { self }.")

    # do the magic to extract the key from the underlying h5-file
    # - for leaves return the numpy array
    # - for inner nodes return a new instance of the class with
    #      the new_path = join(current_path, key) - refering to self.opath
    def __getitem__(self, key: str):

        nodes, leaves = get_nodes(self.file, self.opath)
        return self._extract_value(key, nodes, leaves)

    def keys(self):
        nodes, leaves = get_nodes(self.file, self.opath)
        return set(list(nodes.keys()) + list(leaves.keys()))

    def items(self):
        """
        Note that items will extract all values of terminal nodes (arrays)
        into memory at the current level

        This is not memory efficient! Avoid!
        """
        nodes, leaves = get_nodes(self.file, self.opath)
        return [
            (k, self._extract_value(k, nodes, leaves))
            for k in set(list(nodes.keys()) + list(leaves.keys()))
        ]

    def values(self):
        """
        Note that this will extract all values of terminal nodes (arrays)
        into memory at the current level

        This is not memory efficient! Avoid!
        """
        nodes, leaves = get_nodes(self.file, self.opath)
        return [
            self._extract_value(k, nodes, leaves)
            for k in set(list(nodes.keys()) + list(leaves.keys()))
        ]

    def load(self):
        """
        convert instance of this class to a dictionary - fully loads all descendants recursively

        :return: dictionary containing all descendants of the instance of this class recursively
        """
        nodes, leaves = get_nodes(self.file, self.opath)
        return {
            k: self._extract_value(k, nodes, leaves, recursive=True)
            for k in set(list(nodes.keys()) + list(leaves.keys()))
        }

    def _as_dict(self, slice_length=10, full_load=False):
        """
        Create a dictionary from .h5 file abstraction creating string representations of leaf nodes and slicing arrays and strings

        :param slice_length: length of slices used to represent arrays in leaf nodes as str and twice the length is used for slicing strings
        :param full_load: if True reads the entire array from the underlying as :class:`numpy.ndarray`
        """

        def beautify_string(string: str) -> str:
            # double the array length and both slices from the start -> 4
            return (
                f"{string[:slice_length*4]} ..."
                if len(string) > slice_length * 4
                else string
            )

        def beautify_string_array(arr: np.ndarray) -> np.ndarray:
            shape = arr.shape
            return np.array([beautify_string(v) for v in arr.reshape(-1)]).reshape(
                *shape
            )

        nodes, leaves = get_nodes(self.file, self.opath)

        leaf_nodes = []
        for k, v in leaves.items():
            if full_load:
                leaf_nodes.append((k, v.read()))
            else:
                # determining (byte: S, unicode: U) strings in numpy: '[=<>|](S|U)[0-9]+' where [0-9]+ is length of longest string, [=<>|] byte order
                if re.fullmatch("[=<>|](S|U)[0-9]+", v.dtype.str) and len(v) == 1:
                    x = v[0]
                    if "S" in v.dtype.str:
                        x = v[0].decode()
                    leaf_nodes.append((k, beautify_string(x)))

                elif len(v) > 2 * slice_length:
                    left_slice = v[:slice_length]
                    right_slice = v[-slice_length:]

                    if re.fullmatch("[=<>|](S|U)[0-9]+", v.dtype.str):

                        if "S" in v.dtype.str:
                            left_slice = left_slice.astype(dtype=str)
                            right_slice = right_slice.astype(dtype=str)

                        left_slice = beautify_string_array(left_slice)
                        right_slice = beautify_string_array(right_slice)

                    leaf_nodes.append(
                        (
                            k,
                            f"array({left_slice}"[:-1]
                            + " ... "
                            + f"{right_slice}"[1:]
                            + f") {v.shape} dtype:{left_slice.dtype.name}",
                        )
                    )

                else:

                    v = v.read()

                    if re.fullmatch("[=<>|](S|U)[0-9]+", v.dtype.str):

                        if "S" in v.dtype.str:
                            v = v.astype(dtype=str)

                        v = beautify_string_array(v)

                    leaf_nodes.append((k, f"array({v}) {v.shape} dtype:{v.dtype.name}"))

        return dict(
            [(k, self[k]._as_dict(full_load=full_load)) for k in nodes.keys()]
            + leaf_nodes
        )

    def full_load(self):
        return self._as_dict(full_load=True)

    def __repr__(self):
        def replace_rec(x: dict, find: str, repl: str):
            msg = (
                "value not a str scalar or dict, or Iterable. Call on dictionaries containing str objects"
                + " only as values and keys:\ntype DICT := Union[DICT, Union[Iterable[str],str]]."
                + "\n ie. Nodes are dicts and leaves are Iterables over str or str"
            )
            pairs = []
            for k, v in x.items():
                if isinstance(v, str):
                    pairs.append((k, v.replace(find, repl)))
                elif isinstance(v, dict):
                    pairs.append((k, replace_rec(v, find, repl)))
                elif isinstance(v, Iterable):
                    items = []
                    for e in v:
                        if not isinstance(e, str):
                            raise ValueError(msg)
                        items.append(e.replace(find, repl))
                    pairs.append((k, items))

                else:
                    raise ValueError(msg)

            return dict(pairs)

        # simple repr equivalent to python dict
        # self._as_dict().__repr__()

        # hack to leverage json enc/dec for pretty formatting with indent
        # mask ' in str representation of string type values
        # then create repr of entire dict, replace ' by " (json format - double quotes for keys),
        #      convert to json and unmask
        x = self._as_dict()
        x = replace_rec(x, find="'", repl="~")

        return json.dumps(json.loads(x.__repr__().replace("'", '"')), indent=2).replace(
            "~", "'"
        )


class Writer(Reader):
    """
    Implements a Mutable Mapping Interface for the passed h5 file enabling indexing by key,
    setting key value pairs as well as deleting key value pairs
    Leaf nodes are stored as :class:`numpy.ndarray`.
    """

    def __init__(self, file: tables.file.File, object_path: str):
        """
        Intiializes :class:`Writer` object_path is created in file if not already contained

        :param file: representation of the underlying h5 file
        :param object_path: path within the hdf5 file starting with root '/', eg. '/run_x/data'
        """

        # create object_path in file if does not exist - unfortunately no such publicly exposed fct
        if not file.__contains__(object_path):
            self._create_opath(file, object_path)

        super().__init__(file, object_path)

    def _create_opath(self, file: tables.file.File, object_path: str):
        cpath = "/"
        error = opath.verify(object_path, path_type="abs_path")
        if error:
            raise ValueError(error)
        for comp in opath.split(object_path)[1:]:
            if comp not in file.list_nodes(cpath):
                file.create_group(cpath, comp)
            cpath = opath.join(cpath, comp)

    def __delitem__(self, key: str):
        nodes, leaves = get_nodes(self.file, self.opath)
        if key not in list(nodes.keys()) + list(leaves.keys()):
            raise KeyError(f"Key { key } not contained in { self }.")
        self.file.remove_node(opath.join(self.opath, key), recursive=True)

    # do the magic to create respective array in __set_item__():
    #           test for class then use key to assign appropriately
    #
    # indexing just like nested dictionary - think of it as a file system tree with nodes/directores Node
    #                                                     and leaves/files arrays (ndarray, Array, VArray)
    # use np.ndArray for arrays
    # use Array wrapper to pass further args and kwargs to tables.file.File.create_array()
    # use VArray for variable-length arrays
    # use Node to add another node to the data tree ~ directory in a file system
    # use an arbitrarily nested dict conforming to _Data := Union[Union[np.ndarray, Array, VArray, Node], Dict[str, _Data]]
    def __setitem__(
        self,
        key: str,
        value: Union[np.ndarray, Array, VArray, Node, Dict, List, Tuple, str],
    ):
        # Underlying tables module can only deal with instances that are not void, unicode, or object arrays
        # - in other words if it is castable to a np.ndarray should be fine
        # - especially ragged nesting ([0,[3],2]) and nesting of Iterables of differing lengths ([[0,1],[2,3,4]]) is not possible
        error = opath.verify(key, path_type="single_component")
        if error:
            raise KeyError(f"Keys must conform to following form: {error}")

        # delete node if already exists / overwriting
        if opath.join(self.opath, key) in self.file:
            self.file.remove_node(opath.join(self.opath, key), recursive=True)

        if isinstance(value, Array):
            self.file.create_array(
                self.opath, key, *value.args, obj=value.obj, **value.kwargs
            )
        elif isinstance(value, VArray):
            self.file.create_vlarray(
                self.opath, key, *value.args, obj=value.obj, **value.kwargs
            )
        elif isinstance(value, Node):

            self.file.create_group(self.opath, key)
        elif isinstance(value, Dict):
            # non-tail recursive - allows defining it using magic fcts __delitem__,
            self.file.create_group(self.opath, key)
            node = self[key]
            for nn, vv in value.items():
                # recursion
                node[nn] = vv
        elif isinstance(value, np.ndarray):
            self.file.create_array(self.opath, key, obj=value)
        elif isinstance(value, List) or isinstance(value, Tuple):
            # wrap in np.ndarray to fail early in case of improper nesting as well as dealing with string decoding in Reader
            #   (hdf reads strings in bytes - not as python str objects - unicode)
            self.file.create_array(self.opath, key, obj=np.array(value))

        elif (
            isinstance(value, str)
            or isinstance(value, np.string_)
            or isinstance(value, int)
            or isinstance(value, float)
        ):
            self.file.create_array(self.opath, key, obj=np.asarray([value]))
        else:
            raise ValueError(
                f"No such value type for parameter value supported. Is {type(value)}:\n{value}"
            )


class FileMap:
    """
    Implements Context Manager Interface for :class:`Reader` and  :class:`Writer` which on entering opens the
    h5 file and returns an instance of :class:`Writer` or :class:`Reader` tb used within the context as well as
    closes the h5 file when the context is left

    supported modes are "write"  : open file truncating and read & write
                        "modify" : open file and read & write
                        "read"   : open file in read-only

    indexing just like nested dictionary:
        getting, setting and deleting items is supported
    analagous to file system tree with inner nodes/directories (nodes: Node)
        and leaves/files (arrays: ndarray, Array, VArray)

    Example

    .. testsetup::

        import numpy as np
        from utils import TestEnv
        from persistence import FileMap, Node


    .. testcode::

        with TestEnv():
            with FileMap("file.h5", mode="write") as f:
                f["mydata"] = Node()
                md = f["mydata"]
                md["run_x"] = Node()
                m = md["run_x"]
                m["spikes"] = np.arange(10)
            with FileMap("file.h5", mode="read") as f:
                print(f["mydata"]["run_x"]["spikes"])

    .. testoutput::

        [0 1 2 3 4 5 6 7 8 9]

    file structure:         /mydata/run_x/spikes -> array([...])
    nested dictionary:      {"mydata":{"run_x":{"spikes":array([...])}}}


    Assignment of nested dictionary
    of type XDict := Union[np.ndarray, Dict[str, XDict]]

    .. testsetup::

        import numpy as np
        from utils import TestEnv
        from persistence import FileMap

    .. testcode::

        with TestEnv():
            with FileMap("file.h5", mode="write") as f:
                f["mydata"] = { "run_x" : { "spikes": np.arange(10) }, "array": np.arange(5) }
            with FileMap("file.h5", mode="read") as f:
                print(f)

    .. testoutput::

        {
          "mydata": {
            "run_x": {
              "spikes": "array([0 1 2 3 4 5 6 7 8 9]) (10,) dtype:int64"
            },
            "array": "array([0 1 2 3 4]) (5,) dtype:int64"
          }
        }


    Basic Navigation

    .. testsetup::

        import numpy as np
        from utils import TestEnv
        from persistence import FileMap


    .. testcode::

        with TestEnv():
            with FileMap("file.h5", mode="write") as f:
                f["mydata"] = { "run_x" : { "spikes": np.arange(10) }, "array": np.arange(5) }
            with FileMap("file.h5", mode="read") as f:
                md = f["mydata"]            # move down the object tree / index nested dictionary
                g = md.up()                 # move up the object tree
                print(f.opath)
                print(g.opath)

    .. testoutput::

        /
        /

    """

    def __init__(self, path: str, mode: str = "read", object_path: str = "/"):
        """
        :param path: path to file to be used for persisting
        :param mode: mode of file i/o 'read' (read-only) | 'write' (write, truncating) | 'modify' (write, non-truncating)
        :param object_path: path within the hdf5 file starting with root '/', eg. '/run_x/data'
        """
        error = opath.verify(object_path, path_type="abs_path")
        if error:
            raise ValueError(error)

        if mode == "write":
            path_err = validate_file_path(path, ".h5")
            if path_err:
                raise ValueError(path_err)

            base_path, head = os.path.split(os.path.abspath(path))
            if head in os.listdir(base_path):
                raise ValueError(
                    f"Please provide a unique file name { head } already exists in directory { base_path }."
                )
        elif mode == "modify":
            path_err = validate_file_path(path, ".h5")
            if not os.path.isfile(os.path.abspath(path)) and len(path_err) > 0:
                raise ValueError(
                    f"{path} is not a file nor a valid path for a file: {path_err}."
                )
        elif mode == "read":
            if not os.path.isfile(os.path.abspath(path)):
                raise ValueError(f"No such file found: {path}.")
        else:
            raise ValueError(
                f"Parameter mode may be write | modify | read. Is { mode }."
            )

        self.path = path
        self.mode = mode
        self.opath = object_path

        self.file = None

    # ContextManager Interface
    def __enter__(self):

        # suppress warnings of type tables.NaturalNameWarning - to be able to store invalid path components
        #   (as defined by the tables module, eg. python keywords or str representations of ints)
        #  - this prevents from using these components in member access tables.File.<key>
        #  - which is irrelevant as the tables.File object is wrapped by class Reader or Writer respectively
        warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)

        if self.mode == "write":
            self.file = tables.file.File(self.path, mode="w")
            return Writer(self.file, self.opath)
        elif self.mode == "modify":
            if not os.path.isfile(self.path):
                # create it
                tables.file.File(self.path, mode="w").close()
            self.file = tables.file.File(self.path, mode="r+")
            return Writer(self.file, self.opath)
        else:
            self.file = tables.file.File(self.path, mode="r")
            return Reader(self.file, self.opath)

    def __exit__(self, exc_type, exc_value, traceback):

        if self.file:
            self.file.close()

        if exc_type != None:
            raise exc_type(exc_value, traceback)


def get_nodes(
    file: tables.file.File, object_path: str
) -> Tuple[
    Dict[str, tables.group.Group],
    Dict[str, Union[tables.array.Array, tables.vlarray.VLArray]],
]:
    """
    Retrieves nodes and leaves attached below the node specified by the arg path in the
    object tree of the arg file object

    :param file: representation of the underlying h5 file
    :param path: path to the current node within the object tree
    :return: a list of :class:`Node` and a list of arrays :class:`np.ndarray`, :class:`Array`,
            :class:`VlArray` attached at the current path
    """
    nodes = {}
    leaves = {}
    for node in file.list_nodes(object_path):
        if isinstance(node, tables.group.Group):
            nodes[node._v_name] = node
        elif isinstance(node, tables.array.Array) or isinstance(
            node, tables.vlarray.VLArray
        ):
            leaves[node.name] = node
        else:
            raise Exception(f"No such type supported {type(node)}. Received {node}.")
    return nodes, leaves
