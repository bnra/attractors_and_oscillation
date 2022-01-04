import numpy as np
from brian2 import ms
from attractor import compute_conductance_scaling

from test.utils import TestCase
from BrianExperiment import BrianExperiment
from persistence import FileMap
from network import NeuronPopulation, Connector

from analysis import ExperimentAnalysis, gaussian_smoothing

from differential_equations.neuron_equations import eqs_P, PreEq_AMPA
from differential_equations.neuron_parameters import delay_AMPA
from utils import TestEnv


class TestClassFctComputeConductanceScaling(TestCase):
    @staticmethod
    def _naive_compute_conductance(patterns:np.ndarray, sparsity:float, g_ee:float, c_ee:float):
        size = patterns.shape[1]
        
        g = np.zeros(size * size).reshape(size,size)
        for i in range(size):
            for j in range(size):
                for p in range(patterns.shape[0]):
                    delta_g = g_ee / c_ee * (patterns[p][i] / sparsity - 1) * (patterns[p][j] / sparsity - 1)
                    g[i][j] = max(0, g[i][j] + delta_g)
        return g

    def test_when_called_should_produce_similar_result_to_naiv_impl_when_scaled(self):
        g_ee = 5e-8
        c_ee = 1

        sparsity = 0.1

        size = 20
        p = 100 

        patterns = np.random.choice([True, False], p=[sparsity, 1.0 - sparsity], size=p*size).reshape(p, size)

        s = compute_conductance_scaling(patterns, sparsity)

        g = g_ee/c_ee * s

        g_should = self.__class__._naive_compute_conductance(patterns, sparsity, g_ee, c_ee)

        self.assertTrue(np.allclose(g, g_should))