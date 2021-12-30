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


from BrianExperiment import BrianExperiment
from persistence import FileMap
from network import NeuronPopulation, Connector
from utils import TestEnv
import numpy as np
from differential_equations.neuron_equations import eqs_P, PreEq_AMPA
from differential_equations.neuron_parameters import delay_AMPA
import json
from brian2.units.fundamentalunits import get_unit
import importlib
import brian2

from brian2 import run, ms, second, StateMonitor, Network


with TestEnv():
    with BrianExperiment(
        persist=True,
        path="file.h5",
        neuron_eqs=["eqs_P", "PreEq_AMPA"],
        neuron_params=["delay_AMPA"],
    ) as exp:

        E = NeuronPopulation(
            10, eqs_P, threshold="v_s>-30*mV", refractory=1.3 * ms, method="rk4"
        )
        connect = Connector(synapse_type="static")
        syn_pp = connect(
            E,
            E,
            E.ids,
            E.ids,
            connect=("bernoulli", {"p": 0.3}),
            on_pre=PreEq_AMPA,
            delay=delay_AMPA,
        )

        E.monitor(E.ids, ["v_s"])
        E.monitor_spike(E.ids)
        E.monitor_rate()

        exp.run(5 * ms)

        print(E.monitored)
