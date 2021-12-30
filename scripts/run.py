import sys
import os

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


import argparse
from brian2 import ms, mV, kHz
import brian2
import numpy as np

from BrianExperiment import BrianExperiment
from persistence import FileMap
from network import NeuronPopulation, Connector, PoissonDeviceGroup
from utils import TestEnv
from distribution import draw_uniform
from differential_equations.neuron_equations import (
    eqs_P,
    eqs_I,
    PreEq_AMPA,
    PreEq_AMPA_pois,
    PreEq_GABA,
)
from differential_equations.neuron_parameters import delay_AMPA, delay_GABA, eL_p, eL_i
from utils import generate_sequential_file_name


def run_exp(sim_time, report: bool, progress_bar: bool):

    file_path = generate_sequential_file_name(os.path.join(path, "data"), "exp", ".h5")

    with BrianExperiment(
        report_progress=report,
        progress_bar=progress_bar,
        persist=True,
        path=file_path,
        # object_path="/run_x/data",
        # neuron_eqs=["eqs_P", "PreEq_AMPA"],
        # neuron_params=["delay_AMPA"],
    ) as exp:

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

        S_E_E = connect(
            E,
            E,
            E.ids,
            E.ids,
            connect=("bernoulli", {"p": 0.01}),
            on_pre=PreEq_AMPA,
            delay=delay_AMPA,
        )
        # S_E_E.monitor(S_E_E.synapses, ["x_AMPA"])
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

        E.monitor(E.ids, ["v_s"])
        E.monitor_spike(E.ids)
        E.monitor_rate()

        I.set_pop_var(
            variable="v",
            value=draw_uniform(a=eL_i / mV - 5.0, b=eL_i / mV + 5.0, size=I.size) * mV,
        )

        I.monitor(I.ids, ["v"])
        I.monitor_spike(I.ids)
        I.monitor_rate()

        # external inputs

        # poisson device groups
        PE = PoissonDeviceGroup(size=E.size, rate=1 * kHz)
        PE.monitor_spike(PE.ids)

        PI = PoissonDeviceGroup(size=I.size, rate=1 * kHz)
        PI.monitor_spike(PI.ids)

        # synapses
        S_PE_E = connect(
            PE,
            E,
            PE.ids,
            E.ids,
            connect=("one2one", {}),
            on_pre=PreEq_AMPA_pois,
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

    with FileMap(file_path, mode="read") as f:
        print(f["meta"])


def run(
    sim_time: float,
    report: bool,
    progress_bar: bool,
    virtual: bool,
    deterministic: bool,
):
    if deterministic:
        # note that scipy.stats uses np.random under the hood - used in distribution module
        np.random.seed(0)
        brian2.devices.device.seed(seed=0)
    if virtual:
        with TestEnv():
            run_exp(sim_time, report=report, progress_bar=progress_bar)
    else:
        run_exp(sim_time, report=report, progress_bar=progress_bar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation")

    parser.add_argument(
        "--sim_time",
        type=float,
        default=1000.0,
        help="set simulation time in ms (pot. decimal)",
    )

    parser.add_argument("--report", action="store_true", help="report progress")

    parser.add_argument(
        "--virtual", action="store_true", help="execute in virtual environment"
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="set seed to make all random processes deterministic",
    )

    parser.add_argument(
        "--progress_bar",
        action="store_true",
        help="show progress bar during simulation - may slow execution",
    )

    args = parser.parse_args()
    report = args.report
    progress_bar = args.progress_bar
    virtual = args.virtual
    deterministic = args.deterministic
    sim_time = args.sim_time

    run(
        sim_time,
        report=report,
        progress_bar=progress_bar,
        virtual=virtual,
        deterministic=deterministic,
    )
