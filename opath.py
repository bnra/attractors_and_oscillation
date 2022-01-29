"""
Functions for transforming object paths (path-like strings)
"""

import re
from typing import List


class OpathError(Exception):
    pass


def verify(opath: str, path_type: str = "abs_path") -> str:
    """
    verify path in object tree of hdf5 file

    :param opath: object path in object tree of hd5 file
    :param path_type: path type tb verified choose from single_component | abs_path | rel_path | any_path,
                      where any_path is the superset of the other options, abs_path and rel_path are supersets of single_component paths
    :return: error msg, empty if valid
    """
    base_msg = f"'{opath}' does not conform to path_type = "
    if path_type == "single_component":
        return (
            ""
            if re.fullmatch("[a-zA-Z0-9_-]+", opath)
            else f"{base_msg}single_component: [a-zA-Z0-9_-]+"
        )
    elif path_type == "abs_path":
        return (
            ""
            if re.fullmatch("/(([a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+)?", opath)
            else f"{base_msg}abs_path: /(([a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+)?"
        )
    elif path_type == "rel_path":
        return (
            ""
            if re.fullmatch("([a-zA-Z0-9_]+/)*[a-zA-Z0-9_-]+", opath)
            else f"{base_msg}rel_path: ([a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+"
        )
    elif path_type == "any_path":
        return (
            ""
            if re.fullmatch("[/]?([a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+|/", opath)
            else f"{base_msg}any_path: [/]?([a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+|/"
        )
    else:
        raise OpathError(
            f"Unknown path_type {path_type}. Possible options: single_component | abs_path | rel_path | any_path ."
        )


def split(opath: str) -> List[str]:
    """
    split path in object tree of hdf5 file into path components
    will append root component '/' for absolute paths

    :param opath: object path in object tree of hd5 file
    :return: list of object path components
    """
    comps = [e for e in opath.split("/") if len(e) > 0]
    return ["/"] + comps if opath.startswith("/") else comps


def join(path: str, head: str, *args: str) -> str:
    """
    join arbitrarily many path components in the object tree
    (at least two)

    :param path: base path in the object tree
    :param head: single path component
    :param args: (opt.) further single path components
    :return:     compound path
    """
    additional_comps = "/" + "/".join(args) if len(args) > 0 else ""
    return (
        f"{path}/{head}" + additional_comps
        if not path.endswith("/")
        else path + head + additional_comps
    )
