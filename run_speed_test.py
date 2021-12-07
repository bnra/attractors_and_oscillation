#!/usr/bin/env python
import argparse
from typing import List, Callable
import importlib
import sys
import os
import inspect

from test.utils import SpeedTest, SpeedTester
from utils import format_duration_ns

"""
Example usage script

./run_speed_test.py --test bernoulli.BrianBernoulli bernoulli.ConnectivityBernoulli


Example usage recursive_crawl 

from test.speedtest.run_speed_test import diff_path, path_split, recursive_crawl, flatten_dict
import os

x = recursive_crawl([os.path.abspath("test/speedtest/")], os.path.abspath("."))
tmp = x["test"]["speedtest"]
y = flatten_dict([(tmp,"")])
for fname,fobj in y.items():
    print(fname)
    fobj()

#example2

gather_speed_tests()

"""


def list_path_tree(path):
    files = []
    file_stamps = []
    dirs = []
    dir_stamps = []
    with os.scandir(path) as tree:
        for entry in tree:
            if entry.is_file():
                files.append(entry.name)
                file_stamps.append(entry.stat().st_mtime)
            elif entry.is_dir():
                dirs.append(entry.name)
                dir_stamps.append(entry.stat().st_mtime)
    return files, dirs, file_stamps, dir_stamps


def path_split(path: str):
    comps = []
    # relative paths end on head="relative root" path="", abs paths end on head="" path="/" (for split("/"))
    while True:
        path, head = os.path.split(path)
        if head != "":
            comps.append(head)
        if path == "" or head == "":
            break
    return comps[::-1]


def diff_path(path: str, base_path: str):
    path_comps = path_split(path)
    base_comps = path_split(base_path)
    if len(path_comps) < len(base_comps) or not all(
        [path_comps[i] == b for i, b in enumerate(base_comps)]
    ):
        raise ValueError(f"{base_path} is not a base path of {path}")
    return path_comps[len(base_comps) :]


def recursive_crawl(dirs: List[str], root_path: str, tests: dict = {}):
    if len(dirs) == 0:
        return tests

    path = dirs.pop(0)

    keys = diff_path(path, root_path)

    files, directories, _, _ = list_path_tree(path)

    dirs = dirs + [os.path.join(path, d) for d in directories if d != "__pycache__"]

    module_path = ".".join(keys)

    test_units = {}
    for f in files:
        if not f.endswith(".py") or f.startswith("__"):
            continue
        else:
            f = f[:-3]
        fpath = f"{module_path}.{f}" if len(module_path) > 0 else f

        module = importlib.import_module(fpath)

        tests_f = {}
        for k in dir(module):
            cl = getattr(module, k)
            if (
                inspect.isclass(cl)
                and issubclass(cl, SpeedTest)
                and not inspect.isabstract(cl)
            ):
                if f not in test_units.keys():
                    test_units[f] = {}
                test_units[f][k] = cl

    if len(test_units.keys()) > 0:
        tmp = tests
        for k in keys:
            if k not in tmp:
                tmp[k] = {}
            tmp = tmp[k]
        for fle, funcs in test_units.items():
            tmp[fle] = {}
            for fname, fobj in funcs.items():
                tmp[fle][fname] = fobj

    return recursive_crawl(dirs, root_path, tests)


def flatten_dict(dicts: list, result: dict = {}) -> dict:
    if len(dicts) == 0:
        return result
    d, base = dicts.pop(0)
    for k in d.keys():
        cname = base + "." + k if len(base) > 0 else k
        if isinstance(d[k], dict):
            if len(d[k].keys()) > 0:
                dicts.append((d[k], cname))
            else:
                continue
        else:
            result[cname] = d[k]
    return flatten_dict(dicts, result)


def gather_speed_tests():

    tests = recursive_crawl([os.path.abspath("test/speedtest/")], os.path.abspath("."))

    if len(tests.keys()) == 0:
        return {}

    tests = tests["test"]["speedtest"]

    return flatten_dict([(tests, "")])


def run_tests(
    tests: List,
    trials: int,
    iterations: int,
    reduce_iterations: Callable[[List], float],
    reduce_trials: Callable[[List], float],
):

    n = len(tests)
    print(f"\n\nRunning {n} speed tests")
    for i, (name, test) in enumerate(tests):
        print(f"\nStarting test {name} ({i+1}/{n}):")

        if hasattr(test, "trials"):
            trials = test.trials
        if hasattr(test, "iterations"):
            iterations = test.iterations

        tester = SpeedTester(test=test, reduce=reduce_iterations)
        result = tester.run(trials=trials, iterations=iterations)
        print(f"Result {name}: {format_duration_ns(int(reduce_trials(result)))}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run speed tests")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test",
        nargs="+",
        help="<[[subdir.]*file.]test_func_name>,\n   "
        + "provided string will be suffixed matched against module path of available test cases\n   "
        + "where test_func_name is a name of a function in file file in sub directory subdir of test/speed_test/",
    )
    parser.add_argument(
        "--trials_df",
        type=int,
        default=7,
        help="set default number of trials - used when none specified by respective test case",
    )
    parser.add_argument(
        "--iters_df",
        type=int,
        default=1000,
        help="set default number of iterations per trial- used when none specified by respective test case",
    )
    parser.add_argument(
        "--reduce_trials",
        choices=["min", "avg", "max", "id"],
        default="min",
        type=str,
        help="specify how to reduce over trials for reporting - id to return the list of trial values",
    )
    parser.add_argument(
        "--reduce_iters",
        choices=["min", "avg", "max"],
        default="avg",
        type=str,
        help="specify how to reduce over iterations for reporting",
    )
    group.add_argument(
        "--list", action="store_true", help="list all available test cases"
    )

    args = parser.parse_args()

    trials = args.trials_df
    iterations = args.iters_df
    reduce_iters = args.reduce_iters
    reduce_trials = args.reduce_trials

    reduce = {
        "min": lambda x: min(x),
        "max": lambda x: max(x),
        "avg": lambda x: sum(x) / len(x),
        "id": lambda x: x,
    }

    test_names = args.test

    lst = args.list

    candidates = gather_speed_tests()

    # mutually exclusive with --test option, lst == True -> tests == []
    if lst:
        print("Candidates for speed testing")
        for c in candidates.keys():
            print(f"-{c}")
        sys.exit(0)

    tests = []

    for t in test_names:
        if t in candidates.keys():
            tests.append((t, candidates[t]))
        else:
            potentials = [c for c in candidates.keys() if c.endswith(t)]
            if len(potentials) == 1:
                tests.append((potentials[0], candidates[potentials[0]]))
            elif len(potentials) == 0:
                print(f"No such test {t} found.")
                sys.exit(1)
            else:
                print(
                    f"More than one speed test found (suffix-) matching {t}.\nPlease specify unique suffix of the desired test:"
                )
                for c in potentials:
                    print(f"- {c}")
                sys.exit(1)

    run_tests(
        tests,
        trials=trials,
        iterations=iterations,
        reduce_iterations=reduce[reduce_iters],
        reduce_trials=reduce[reduce_trials],
    )
