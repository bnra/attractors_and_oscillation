import numpy as np
import tables

import os
import re
from typing import Tuple, Union, Dict

from utils import validate_file_path

class Array:
    """
    Placeholder for tables.array.Array()
    used by Reader and Writer class to enable Mapping Interface while also allowing arbitrary
    nesting 
    """
    def __init__(self, obj:np.ndarray, *args, **kwargs):
        self.obj =obj
        self.args = args
        self.kwargs = kwargs

class VArray:
    """
    Placeholder for tables.vlarray.VlArray()
    used by Reader and Writer class to enable Mapping Interface while also allowing arbitrary
    nesting
    """    
    def __init__(self, *args, obj:Union[np.ndarray, None] = None, **kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs

class Node:
    """
    Placeholder for tables.groups.Group()
    used by Reader and Writer class to enable Mapping Interface while also allowing arbitrary
    nesting 
    """
    def __init__(self):
        pass


class Reader:
    """
    Implements a Mapping Interface for the passed h5 file enabling indexing by key
    """
    def __init__(self, file:tables.file.File, opath:str):
        self.opath = opath
        self.file = file

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        for node in self.keys(): 
            yield node

    def extract_value_(self, key:str, nodes:Tuple[Dict[str, tables.group.Group]], leaves:\
            Dict[str, Union[tables.array.Array, tables.vlarray.VLArray]]):
        if key in nodes.keys():
            return self.__class__(self.file, join_opath(self.opath, key))
        else:
            if key in leaves.keys():
                return leaves[key].read()
            else:
                raise KeyError(f"Key { key } not contained in { self }.")

    # do the magic to extract the key from the underlying h5-file 
    # - for leaves return the numpy array
    # - for inner nodes return a new instance of the class with 
    #      the new_path = join(current_path, key) - refering to self.opath   
    def __getitem__(self, key:str):
        nodes, leaves = get_nodes(self.file, self.opath)
        return self.extract_value_(key, nodes, leaves)

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
        return [(k,self.extract_value_(k, nodes, leaves)) \
            for k in set(list(nodes.keys()) + list(leaves.keys()))]


class Writer(Reader):
    """
    Implements a Mutable Mapping Interface for the passed h5 file enabling indexing by key,
        setting key value pairs as well as deleting key value pairs
    """
    def __init__(self, file:tables.file.File, opath:str):
        super().__init__(file, opath)

    
    def __delitem__(self, key:str):
        nodes, leaves = get_nodes(self.file, self.opath)
        if key not in list(nodes.keys()) + list(leaves.keys()):
            raise KeyError(f"Key { key } not contained in { self }.")
        self.file.remove_node(join_opath(self.opath, key), recursive=True)

    # do the magic to create respective array in __set_item__(): 
    #           test for class then use key to assign appropriately
    #
    # indexing just like nested dictionary - think of it as a file system tree with nodes/directores Node 
    #                                                     and leaves/files arrays (ndarray, Array, VArray)
    # use np.ndArray for arrays
    # use Array wrapper to pass further args and kwargs to tables.file.File.create_array()
    # use VArray for variable-length arrays
    # use Node to add another node to the data tree ~ directory in a file system
    def __setitem__(self, key:str, value:Union[np.ndarray, Array, VArray, Node]):
        if isinstance(value, np.ndarray):
            self.file.create_array(self.opath, key, obj=value)
        elif isinstance(value, Array):
            self.file.create_array(self.opath, key, *value.args, obj=value.obj, **value.kwargs)
        elif isinstance(value, VArray):
            self.file.create_vlarray(self.opath, key, *value.args, obj=value.obj, **value.kwargs)
        else:
            self.file.create_group(self.opath, key)

class FileMap:
    """
    Implements Context Manager Interface for class Reader and Writer which on entering opens the
    h5 file and returns an instance of Writer or Reader tb used within the context as well as
    closes the h5 file when the context is left 

    supported modes are "write"  : open file truncating and read & write
                        "modify" : open file and read & write
                        "read"   : open file in read-only  

    indexing just like nested dictionary:
        getting, setting and deleting items is supported 
    analagous to file system tree with inner nodes/directories (nodes: Node) 
        and leaves/files (arrays: ndarray, Array, VArray)

    >>>
    with FileMap("some/path/file.h5", writable=True) as f:
        f["mydata"] = Node()     
        md = f["mydata"]
        md["run_x"] = Node()
        m = md["run_x"]
        m["spikes"] = np.random.choice([0,1], size=100, p=[0.99, 0.01])
        print(m["spikes"])
    <<<
    file structure:         /mydata/run_x/spikes -> np.ndarray
    nested dictionary:      {"mydata":{"run_x":{"spikes":np.ndarray}}}
    """

    def __init__(self, path:str, mode:str="read"):
        
        if mode == "write":
            path_err = validate_file_path(path, ".h5")
            if path_err:
                raise ValueError(path_err)
        
            base_path, head = os.path.split(os.path.abspath(path))
            if head in os.listdir(base_path):
                raise ValueError(f"Please provide a unique file name { head } already exists in directory { base_path }.")
        elif mode == "modify" or mode == "read":
            if not os.path.isfile(os.path.abspath(path)):
                raise ValueError(f"No such file found: {path}.")
        else:
            raise ValueError(f"Parameter mode may be write | modify | read. Is { mode }.")

        self.path = path
        self.mode = mode

        self.file = None
         
    
    # ContextManager Interface
    def __enter__(self):
        if self.mode == "write":
            self.file = tables.file.File(self.path, mode='w')
            return Writer(self.file, "/")
        elif self.mode == "modify":
            self.file = tables.file.File(self.path, mode='r+')
            return Writer(self.file, "/")
        else:
            self.file = tables.file.File(self.path, mode='r')
            return Reader(self.file, "/")

    def __exit__(self, exc_type, exc_value, traceback):

        if self.file:
            self.file.close()

        if exc_type != None:
            raise exc_type(exc_value, traceback)



def get_nodes(file:tables.file.File, path:str)->Tuple[Dict[str, tables.group.Group],
                            Dict[str, Union[tables.array.Array, tables.vlarray.VLArray]]]:
    """
    Retrieves nodes and leaves attached below the node specified by the arg path in the 
    object tree of the arg file object

    args
    - file              table.File object of the underlying h5 file
    - path              path to the current node within the object tree

    returns
    - Tuple[Dict[str,node], Dict[str, array]]
                        a list of nodes and a list of arrays attached at
                        the current path 
    """
    nodes = {}
    leaves = {}
    for node in file.list_nodes(path):
        if isinstance(node, tables.group.Group):
            nodes[node._v_name] = node
        elif isinstance(node, tables.array.Array) or isinstance(node, tables.vlarray.VLArray):
            leaves[node.name] = node
    return nodes, leaves


def join_opath(path:str, head:str)->str:
    """
    join path components in the object tree
    
    args
    - path          base path in the object tree
    - head          single path component 

    returns         compound path
    """
    return f"{path}/{head}" if not path.endswith("/") else path + head
