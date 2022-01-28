import itertools
from test.utils import SpeedTest

from utils import TestEnv
import mp

import numpy as np
from brian2 import ms, mV, kHz
import os

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


class MultiProcessSim(SpeedTest):

    trials = 1
    iterations = 1

    def run(self):
        base_path = "base_path"
        with TestEnv() as env:
            os.makedirs(os.path.join(env.tmp_dir, base_path))

            pool = mp.Pool(
                base_path,
                {
                    "sparsity": [0.17, 0.23],
                    "num_patterns": [100, 200],
                    "norm": ["norm", "z_score"],
                    "clip": ["clipped_once", "pos_only"],
                },
                run_exp,
                {"scaling": 5.0},
            )
            pool.run()


class SingleProcessSim(SpeedTest):

    trials = 1
    iterations = 1

    def run(self):

        file_name_generator = (
            lambda instance: "_".join(
                [
                    f"{c}_{mp.float_to_path_component(v)}"
                    if isinstance(v, float)
                    else f"{c}_{v}"
                    for c, v in instance
                ]
            )
            + ".h5"
        )

        base_path = os.path.abspath("base_path")

        scaling = 5.0
        sparsity = [0.17, 0.23]
        num_patterns = [100, 200]
        norm = ["norm", "z_score"]
        clip = ["clipped_once", "pos_only"]
        with TestEnv() as env:
            os.makedirs(os.path.join(env.tmp_dir, base_path))

            for s, p, n, c in itertools.product(sparsity, num_patterns, norm, clip):
                fname = file_name_generator(
                    [("sparsity", s), ("num_patterns", p), ("norm", n), ("clip", c)]
                )
                run_exp(os.path.join(base_path, fname), s, p, n, c, scaling)
