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


def run_exp_zscore_sd(
    path: str,
    sparsity: float,
    numpatterns: int,
    negoffsetsd: float,
    esize: int,
    rpe: float,
    rpi: float,
):

    sim_time = 500.0

    with BrianExperiment(persist=True, path=path, report_progress=True) as exp:

        # EI network

        # populations
        E = NeuronPopulation(
            esize, eqs_P, threshold="v_s>-30*mV", refractory=1.3 * ms, method="rk4"
        )
        I = NeuronPopulation(
            esize // 4, eqs_I, threshold="v>-30*mV", refractory=1.3 * ms, method="rk4"
        )

        # synapses
        connect = Connector(synapse_type="static")

        # patterns
        patterns = np.random.choice(
            [True, False], p=[sparsity, 1.0 - sparsity], size=numpatterns * esize
        ).reshape(numpatterns, esize)

        # compute unclipped scaling
        s = attractor.compute_conductance_scaling_unclipped(patterns, sparsity)

        s = attractor.z_score(s)
        # - negoffset - note (0,1)- gaussian
        s = s - negoffsetsd

        s = np.maximum(0, s)

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
        PE = PoissonDeviceGroup(size=E.size, rate=rpe * kHz)
        # PE.monitor_spike(PE.ids)

        PI = PoissonDeviceGroup(size=I.size, rate=rpi * kHz)
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


def run_exp(
    path: str,
    sparsity: float,
    numpatterns: int,
    norm: str,
    scaling: int,
    esize: int,
    rpe: float,
    rpi: float,
):

    sim_time = 500.0

    normalizations = {
        "id": lambda x: x,
        "norm": attractor.normalize,
        "zscore": attractor.z_score,
    }

    with BrianExperiment(persist=True, path=path) as exp:

        # EI network

        # populations
        E = NeuronPopulation(
            esize, eqs_P, threshold="v_s>-30*mV", refractory=1.3 * ms, method="rk4"
        )
        I = NeuronPopulation(
            esize // 4, eqs_I, threshold="v>-30*mV", refractory=1.3 * ms, method="rk4"
        )

        # synapses
        connect = Connector(synapse_type="static")

        # patterns

        patterns = np.random.choice(
            [True, False], p=[sparsity, 1.0 - sparsity], size=numpatterns * E.size
        ).reshape(numpatterns, E.size)

        # scale factor

        # s = np.zeros(E.size * E.size).reshape(E.size, E.size)

        # compute unclipped scaling
        s = attractor.compute_conductance_scaling_unclipped(patterns, sparsity)
        s = normalizations[norm](s)
        s = np.maximum(0, s)
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
        PE = PoissonDeviceGroup(size=E.size, rate=rpe * kHz)
        # PE.monitor_spike(PE.ids)

        PI = PoissonDeviceGroup(size=I.size, rate=rpi * kHz)
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

    parser.add_argument(
        "--sim",
        type=bool,
        choices=["simple", "normal", "zscore"],
        help="run brian simulation",
    )

    args = parser.parse_args()

    sim = args.sim

    base_path = "/mnt/idp/zscore_exploration"

    if sim == "normal":
        params = {
            "sparsity": np.arange(0.08, 0.3, 0.01),  # 0.17, 0.23
            "numpatterns": [n for n in range(120, 145, 5)],
            "norm": ["norm", "zscore", "id"],
        }
        kwargs = {"scaling": 5.0, "esize": 4000, "rpe": 6.5, "rpi": 1.0}
        f = run_exp
    elif sim == "simple":
        # simple
        params = {"b": [2, 3, 4], "data": list(range(99990, 100000)), "acc": [1, 2]}
        f = fmap
        kwargs = {}
    else:
        # "zscore"

        params = {
            "sparsity": np.arange(0.08, 0.25, 0.02),
            "numpatterns": [130],
            "negoffsetsd": [0.8 + e for e in np.arange(0.0, 0.5, 0.1)],
            "rpe": np.arange(1.0, 8.0, 1.0),
            "rpi": np.arange(1.0, 8.0, 1.0),
        }
        kwargs = {
            "esize": 4000,
        }
        f = run_exp_zscore_sd

    pool = mp.Pool(
        base_path,
        params,
        f,
        kwargs,
    )
    pool.run()
