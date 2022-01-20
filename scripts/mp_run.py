import sys
import os
import argparse

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

from utils import TestEnv

import mp

import numpy as np
from brian2 import ms, mV, kHz
import os
import multiprocessing

import attractor
from utils import TestEnv

from BrianExperiment import BrianExperiment
from network import NeuronPopulation, PoissonDeviceGroup, Connector
from distribution import draw_uniform
from differential_equations.neuron_equations import (
    eqs_P,
    eqs_I,
    PreEq_AMPA,
    PreEq_AMPA_pois,
    PreEq_GABA,
)
from differential_equations.neuron_parameters import delay_AMPA, delay_GABA, eL_p, eL_i


def run_exp(
    path: str, sparsity: float, num_patterns: int, norm: str, clip: str, scaling: int
):

    sim_time = 500.0
    rate_poisson_e = 6.5
    rate_poisson_i = 1.0

    clippings = {
        "clipped_once": attractor.compute_conductance_scaling_single_clip,
        "unclipped": attractor.compute_conductance_scaling_unclipped,
        "pos_only": attractor.compute_conductance_scaling,
    }
    normalizations = {
        "id": lambda x: x,
        "norm": attractor.normalize,
        "z_score": attractor.z_score,
    }

    with BrianExperiment(persist=True, path=path) as exp:

        # EI network

        # populations
        E = NeuronPopulation(
            4000, eqs_P, threshold="v_s>-30*mV", refractory=1.3 * ms, method="rk4"
        )
        I = NeuronPopulation(
            1000, eqs_I, threshold="v>-30*mV", refractory=1.3 * ms, method="rk4"
        )

        # synapses
        connect = Connector(synapse_type="static")

        # patterns

        patterns = np.random.choice(
            [True, False], p=[sparsity, 1.0 - sparsity], size=num_patterns * E.size
        ).reshape(num_patterns, E.size)

        # scale factor

        # s = np.zeros(E.size * E.size).reshape(E.size, E.size)

        # compute unclipped scaling
        s = clippings[clip](patterns, sparsity)
        s = normalizations[norm](s)
        s = s * scaling

        S_E_E = connect(
            E,
            E,
            E.ids,
            E.ids,
            connect=("all2all", {}),
            model="scale : 1",
            on_pre="x_AMPA += alphax * scale",
            delay=delay_AMPA,
            syn_params={"scale": s},
        )

        # S_E_E.monitor(S_E_E.synapses[0:10], variables=["x_AMPA"])

        S_E_I = connect(
            E,
            I,
            E.ids,
            I.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=PreEq_AMPA,
            delay=delay_AMPA,
        )

        S_I_E = connect(
            I,
            E,
            I.ids,
            E.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=PreEq_GABA,
            delay=delay_GABA,
        )

        S_I_I = connect(
            I,
            I,
            I.ids,
            I.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=PreEq_GABA,
            delay=delay_GABA,
        )

        # initialize vars and monitor

        E.set_pop_var(
            variable="v_s",
            value=draw_uniform(a=eL_p / mV - 5.0, b=eL_p / mV + 5.0, size=E.size) * mV,
        )
        E.set_pop_var(
            variable="v_d",
            value=draw_uniform(a=eL_p / mV - 5.0, b=eL_p / mV + 5.0, size=E.size) * mV,
        )

        E.monitor(
            E.ids[0:2], ["v_s", "x_AMPA", "synP", "x_AMPA_ext", "synP_ext"], dt=1.0 * ms
        )
        E.monitor_spike(E.ids)

        I.set_pop_var(
            variable="v",
            value=draw_uniform(a=eL_i / mV - 5.0, b=eL_i / mV + 5.0, size=I.size) * mV,
        )

        I.monitor(I.ids[0:2], ["v"], dt=1.0 * ms)
        I.monitor_spike(I.ids)

        # external inputs

        # poisson device groups
        PE = PoissonDeviceGroup(size=E.size, rate=rate_poisson_e * kHz)
        # PE.monitor_spike(PE.ids)

        PI = PoissonDeviceGroup(size=I.size, rate=rate_poisson_i * kHz)
        # PI.monitor_spike(PI.ids)

        # synapses
        S_PE_E = connect(
            PE,
            E,
            PE.ids,
            E.ids,
            connect=("one2one", {}),
            on_pre="x_AMPA_ext += 1.5*alphax",
            delay=delay_AMPA,
        )
        S_PI_I = connect(
            PI,
            I,
            PI.ids,
            I.ids,
            connect=("one2one", {}),
            on_pre=PreEq_AMPA_pois,
            delay=delay_AMPA,
        )

        exp.run(sim_time * ms)


def fmap(path, b, data, acc):
    proc = multiprocessing.current_process()

    x = np.math.factorial(b * data + acc)
    print(
        f"                    Result {proc.name} w/ id {proc.pid}: {b,data,acc} -> {x.bit_length()}\n\n"
    )
    return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run simulation")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--sim", action="store_true", help="run brian simulation")

    group.add_argument("--simple", action="store_true", help="run simple example")

    args = parser.parse_args()

    sim = args.sim
    simple = args.simple

    base_path = "base_path"

    if sim:
        params = {
            "sparsity": [0.17, 0.23],
            "num_patterns": [100, 200],
            "norm": ["norm", "z_score"],
            "clip": ["clipped_once", "pos_only"],
        }
        kwargs = {"scaling": 5.0}
        f = run_exp
    else:
        # simple
        params = {"b": [2, 3, 4], "data": list(range(99990, 100000)), "acc": [1, 2]}
        f = fmap
        kwargs = {}

    with TestEnv() as env:
        os.makedirs(os.path.join(env.tmp_dir, base_path))

        pool = mp.Pool(
            base_path,
            params,
            f,
            kwargs,
        )
        pool.run()
