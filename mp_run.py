import sys
import os
import argparse
from typing import Callable
import itertools
import tqdm
import re

path = os.path.abspath(".")

# test directory is at the root level
root_checked = False
while "test" not in os.listdir(path):
    path = os.path.dirname(path)
    if path == "/":
        if root_checked:
            print(
                "root path not found - please execute from within the root directory of the repository or any of its (nested) subdirectories"
            )
            sys.exit(1)
        else:
            root_checked = True
sys.path.insert(0, path)


import mp

import numpy as np
import os

from utils import TestEnv

from persistence import FileMap

from experiments_eif import (
    generate_fixed_patterns,
    run_exp_eif_attr,
    run_exp_eif_attr_blocked_stimulus,
)





def run_single_proc(f: Callable, parameters, kwargs, base_path):

    plabels, pvalues = zip(*[(p, parameters[p]) for p in parameters.keys()])
    # instances = itertools.product(*pvalues)

    for instance in tqdm.tqdm(itertools.product(*pvalues)):
        kwg = kwargs.copy()
        instance = list(zip(plabels, instance))
        for p, v in instance:
            kwg[p] = v
        fn_params = []
        for k, v in list(kwargs.items()) + instance:
            if np.isscalar(v):
                fn_params.append((k, v))
            elif isinstance(v, dict) and any([np.isscalar(vv) for vv in v.values()]):
                for kk, vv in v.items():
                    if np.isscalar(vv):
                        fn_params.append((kk, vv))
        fname = mp.file_name_generator(fn_params)
        # [(k, v) for k, v in list(kwargs.items()) + instance if np.isscalar(v)]
        fpath = os.path.join(base_path, fname)

        kwg["path"] = fpath

        f(**kwg)


def run_experiments(
    f: Callable,
    parameters,
    kwargs,
    base_path,
    multi_proc: bool = True,
    num_procs: int = None,
):
    if multi_proc == True:
        pool = mp.Pool(base_path, parameters, f, kwargs, num_procs=num_procs)
        pool.run()
    else:
        run_single_proc(f, parameters, kwargs, base_path)


def parse_cli_arg_iterable(s: str):
    if re.fullmatch("\[[0-9]+(.[0-9]+)?:[0-9]+(.[0-9]+)?:[0-9]+(.[0-9]+)?\]", s):
        return np.arange(
            *[float(r) if "." in r else int(r) for r in s[1:-1].split(":")]
        )
    elif re.fullmatch("[0-9]+(.[0-9]+)?(,[0-9]+(.[0-9]+)?)*", s):
        return [
            float(e) if "." in e else int(e)
            for e in [e for e in s.split(",") if len(e) > 0]
        ]
    else:
        raise ValueError(f"format of string {s} unknown")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run simulation")

    parser.add_argument(
        "--sim",
        type=str,
        choices=[
            "eif_attr",
            "eif_attr_stim",
        ],
        help="run brian simulation",
    )
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--continuous_stim", type=str, choices=["True", "False"], default="False"
    )
    parser.add_argument(
        "--weighted", type=str, choices=["True", "False", "True,False"], default="True"
    )
    parser.add_argument("--simtime", type=float, default=1000.0)
    parser.add_argument(
        "--perturbation",
        type=str,
        default="0.0",
        help="comma separated list or slice ('[start:end:step]') of perturbation values",
    )
    parser.add_argument(
        "--stimulus_pattern_idx",
        type=str,
        default="0",
        help="comma separated list or slice ('[start:end:step]') of indices tb used as pattern for stimulus presentation",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="1.0",
        help="comma separated list or slice ('[start:end:step]') of norm values",
    )

    parser.add_argument(
        "--rpe",
        type=str,
        default="3.9",
        help="comma separated list or slice ('[start:end:step]') of norm values",
    )

    parser.add_argument(
        "--rpi",
        type=str,
        default="5.15",
        help="comma separated list or slice ('[start:end:step]') of norm values",
    )


    parser.add_argument(
        "--beta",
        type=str,
        default="0.5",
        help="comma separated list or slice ('[start:end:step]') of norm values",
    )

    parser.add_argument(
        "--minusbeta",
        type=str,
        default="1.0",
        help="comma separated list or slice ('[start:end:step]') of norm values",
    )

    args = parser.parse_args()

    sim = args.sim
    base_path = args.path

    continuous_stim = True if args.continuous_stim == "True" else False
    weighted = (
        (True,)
        if args.weighted == "True"
        else (True, False)
        if args.weighted == "True,False"
        else (False,)
    )
    perturbation = parse_cli_arg_iterable(args.perturbation)
    norm = parse_cli_arg_iterable(args.norm)
    stimulus_pattern_idx = parse_cli_arg_iterable(args.stimulus_pattern_idx)

    rpe = parse_cli_arg_iterable(args.rpe)
    rpi = parse_cli_arg_iterable(args.rpi)


    beta = parse_cli_arg_iterable(args.beta)
    minusbeta = parse_cli_arg_iterable(args.minusbeta)

    simtime = args.simtime

    if not os.path.isdir(base_path):
        if not os.path.isfile(base_path):
            print(f"Creating directory {base_path}.")
            os.makedirs(base_path)
        else:
            raise ValueError(
                "A file of name {base_path} exists already (yet not a directory.)"
            )

    if sim == "eif_attr":

        pattern_fname = "pattern.h5"
        pattern_path = os.path.abspath(os.path.join(base_path, pattern_fname))

        # generate pattern if file does not exist yet
        if not os.path.isfile(pattern_path):
            # generate pattern with fixed number of 1s ~ allows computing single threshold for all patterns

            # E pop size ~ pattern_length
            esize = 4000
            sparsity = 0.05  # 0.2

            pattern = generate_fixed_patterns(
                esize=esize, sparsity=sparsity, numpatterns=20
            )

            # persist pattern
            with FileMap(os.path.join(base_path, pattern_fname), mode="write") as f:
                f["pattern"] = pattern
                f["sparsity"] = sparsity
        else:
            with FileMap(os.path.join(base_path, pattern_fname), mode="read") as f:
                pattern = f["pattern"]
                sparsity = f["sparsity"]

        esize = pattern.shape[1]

    
        params = {
            "rpe": rpe,  
            "rpi": rpi,
            "weighted": [*weighted],
            "norm": norm,

        }

        kwargs = {
            "esize": esize,
            "simtime": simtime,
            "sparsity": sparsity,
            "pattern": pattern,
        }

        f = run_exp_eif_attr

        run_experiments(f, params, kwargs, base_path, multi_proc=True, num_procs=6)

    elif sim == "eif_attr_stim":

        pattern_fname = "pattern.h5"
        pattern_path = os.path.abspath(os.path.join(base_path, pattern_fname))

        # generate pattern if file does not exist yet
        if not os.path.isfile(pattern_path):
            # generate pattern with fixed number of 1s ~ allows computing single threshold for all patterns

            # E pop size ~ pattern_length
            esize = 4000
            sparsity = 0.05

            pattern = generate_fixed_patterns(
                esize=esize, sparsity=sparsity, numpatterns=20
            )

            # persist pattern
            with FileMap(os.path.join(base_path, pattern_fname), mode="write") as f:
                f["pattern"] = pattern
                f["sparsity"] = sparsity
        else:
            with FileMap(os.path.join(base_path, pattern_fname), mode="read") as f:
                pattern = f["pattern"]
                sparsity = f["sparsity"]

        esize = pattern.shape[1]

        if (
            max(stimulus_pattern_idx) >= pattern.shape[0]
            or min(stimulus_pattern_idx) < 0
        ):
            raise ValueError(
                f"--stimulus_pattern_idx must be in [0, num_patterns-1], num_patterns is {pattern.shape[0]}"
            )

        params = {
            "rpe": rpe,
            "rpi": rpi,
            "weighted": [*weighted],
            "beta": beta, 
            "minusbeta": minusbeta,
            "norm": norm,
            "perturbation": perturbation,
            "stimuluspatternidx": stimulus_pattern_idx,
        }

        kwargs = {
            "esize": esize,
            "simtime": simtime,
            "sparsity": sparsity,
            "pattern": pattern,
            "continuousstim": continuous_stim,
        }

        f = run_exp_eif_attr_blocked_stimulus

        

        run_experiments(f, params, kwargs, base_path, multi_proc=True, num_procs=6)
