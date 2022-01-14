import numpy as np
from attractor import (
    compute_conductance_scaling,
    compute_conductance_scaling_single_clip,
    compute_conductance_scaling_unclipped,
    normalize,
    z_score,
)

from test.utils import TestCase


class TestClassFctComputeConductanceScaling(TestCase):
    @staticmethod
    def _naive_compute_conductance(
        patterns: np.ndarray, sparsity: float, g_ee: float, c_ee: float
    ):
        size = patterns.shape[1]

        g = np.zeros(size * size).reshape(size, size)
        for i in range(size):
            for j in range(size):
                for p in range(patterns.shape[0]):
                    delta_g = (
                        g_ee
                        / c_ee
                        * (patterns[p][i] / sparsity - 1)
                        * (patterns[p][j] / sparsity - 1)
                    )
                    g[i][j] = max(0, g[i][j] + delta_g)
        return g

    def test_when_called_should_produce_similar_result_to_naiv_impl_when_scaled(self):
        g_ee = 5e-8
        c_ee = 1.0

        sparsity = 0.1

        size = 20
        p = 100

        patterns = np.random.choice(
            [True, False], p=[sparsity, 1.0 - sparsity], size=p * size
        ).reshape(p, size)

        s = compute_conductance_scaling(patterns, sparsity)

        g = g_ee / c_ee * s

        g_should = self.__class__._naive_compute_conductance(
            patterns, sparsity, g_ee, c_ee
        )

        self.assertTrue(np.allclose(g, g_should))


class TestClassFctComputeConductanceScalingSingleClip(TestCase):
    @staticmethod
    def _naive_compute_conductance_single_clip(
        patterns: np.ndarray, sparsity: float, g_ee: float, c_ee: float
    ):
        size = patterns.shape[1]

        g = np.zeros(size * size).reshape(size, size)
        for i in range(size):
            for j in range(size):
                for p in range(patterns.shape[0]):
                    g[i][j] += (
                        g_ee
                        / c_ee
                        * (patterns[p][i] / sparsity - 1)
                        * (patterns[p][j] / sparsity - 1)
                    )
                g[i][j] = max(0, g[i][j])
        return g

    def test_when_called_should_produce_similar_result_to_naiv_impl_when_scaled(self):
        g_ee = 5e-8
        c_ee = 1

        sparsity = 0.1

        size = 20
        p = 100

        patterns = np.random.choice(
            [True, False], p=[sparsity, 1.0 - sparsity], size=p * size
        ).reshape(p, size)

        s = compute_conductance_scaling_single_clip(patterns, sparsity)

        g = g_ee / c_ee * s

        g_should = self.__class__._naive_compute_conductance_single_clip(
            patterns, sparsity, g_ee, c_ee
        )

        self.assertTrue(np.allclose(g, g_should))


class TestClassFctComputeConductanceScalingUnclipped(TestCase):
    @staticmethod
    def _naive_compute_conductance_unclipped(
        patterns: np.ndarray, sparsity: float, g_ee: float, c_ee: float
    ):
        size = patterns.shape[1]

        g = np.zeros(size * size).reshape(size, size)
        for i in range(size):
            for j in range(size):
                for p in range(patterns.shape[0]):
                    g[i][j] += (
                        g_ee
                        / c_ee
                        * (patterns[p][i] / sparsity - 1)
                        * (patterns[p][j] / sparsity - 1)
                    )
        return g

    def test_when_called_should_produce_similar_result_to_naiv_impl_when_scaled(self):
        g_ee = 5e-8
        c_ee = 1

        sparsity = 0.1

        size = 20
        p = 100

        patterns = np.random.choice(
            [True, False], p=[sparsity, 1.0 - sparsity], size=p * size
        ).reshape(p, size)

        s = compute_conductance_scaling_unclipped(patterns, sparsity)

        g = g_ee / c_ee * s

        g_should = self.__class__._naive_compute_conductance_unclipped(
            patterns, sparsity, g_ee, c_ee
        )

        self.assertTrue(np.allclose(g, g_should))


class TestClassFctNormalize(TestCase):
    def test_case_interval_above_zero_should_squash_to_0_and_1(self):
        matrix = np.arange(9).reshape(3, 3) + 1
        m_norm = normalize(matrix)
        # ensure step size (exploiting sequence of values with step size 1 in original matrix - np.arange), and max and min
        self.assertTrue(
            np.all(m_norm.ravel()[1:] - m_norm.ravel()[:-1] == 1.0 / (matrix.size - 1))
        )
        self.assertTrue(np.min(m_norm) == 0.0)
        self.assertTrue(np.max(m_norm) == 1.0)

    def test_case_min_below_zero_and_max_above_should_squash_to_0_and_1(self):
        matrix = np.arange(9).reshape(3, 3) - 1
        m_norm = normalize(matrix)
        # ensure step size (exploiting sequence of values with step size 1 in original matrix - np.arange), and max and min
        self.assertTrue(
            np.all(m_norm.ravel()[1:] - m_norm.ravel()[:-1] == 1.0 / (matrix.size - 1))
        )
        self.assertTrue(np.min(m_norm) == 0.0)
        self.assertTrue(np.max(m_norm) == 1.0)

    def test_case_min_below_zero_and_max_below_should_squash_to_0_and_1(self):
        matrix = np.arange(9).reshape(3, 3) - 10
        m_norm = normalize(matrix)
        # ensure step size (exploiting sequence of values with step size 1 in original matrix - np.arange), and max and min
        self.assertTrue(
            np.all(m_norm.ravel()[1:] - m_norm.ravel()[:-1] == 1.0 / (matrix.size - 1))
        )
        self.assertTrue(np.min(m_norm) == 0.0)
        self.assertTrue(np.max(m_norm) == 1.0)

    def test_setting_rescale_interval_should_squash_to_frm_and_to(self):
        frm = 3.0
        to = 10.0

        matrix = np.arange(9).reshape(3, 3) - 1
        m_norm = normalize(matrix, frm=frm, to=to)
        # ensure step size (exploiting sequence of values with step size 1 in original matrix - np.arange), and max and min
        self.assertTrue(
            np.all(
                m_norm.ravel()[1:] - m_norm.ravel()[:-1]
                == (to - frm) / (matrix.size - 1)
            )
        )
        self.assertTrue(np.min(m_norm) == frm)
        self.assertTrue(np.max(m_norm) == to)


class TestClassFctZScore(TestCase):
    def test_when_called_should_return_result_with_mean_0_and_unit_var(self):
        x = np.arange(100)
        z = z_score(x)
        self.assertTrue(np.mean(z) < 1e-15)
        self.assertTrue(np.std(z) - 1.0 < 1e-15)
