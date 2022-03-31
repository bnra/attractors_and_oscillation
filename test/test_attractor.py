import numpy as np
import attractor
from attractor import (
    compute_conductance_scaling,
    compute_conductance_scaling_single_clip,
    compute_conductance_scaling_unclipped,
    p_value_snapshot_same_sparsity,
    similarity_threshold,
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


class TestClassPValueSnapshot(TestCase):
    def test_when_basic_input_provided_should_return_same_as_manually_computed(self):
        sparsity = 0.5
        # we want to test for a 4 component vector with similarity of 2 ie 2 components match
        nflips = 2
        pattern = np.array(
            [True, False, False, True]
        )  # np.random.choice([True, False], p=[sparsity, 1-sparsity], size=size)
        size = pattern.size

        permutation = np.array(
            [False, True, False, True]
        )  # 2 flips: a 1 flip and a 0 flip

        # patterns with at most one 1 flip and at most 1 zero flip -
        # ones:= np.sum(pattern) ie all 1s in pattern
        # as the number of one flips is lowerbounded by max(0, nflips - (size-ones))
        #  and upper bounded by min(ones, nflips) ie we sum over flips 0 or
        #     if nflips - number of 0s in pattern is > 0 then  nflips - number of 0s in pattern (as we cannot flip more 0s than there are in pattern)
        #   to the minimum of ones in pattern (maximal # of flips possible) and nflips (as we compute the sum of at most nflips)
        #  -> lower bound 0 as nflips=2 and ones = 2 -> nflips-ones=0
        #  -> upper bound 2 as ones = nflips
        # no flips: TFFT
        # 1 zero flip (F->T): TTFT, TFTT
        # 1 one flip (T->F): FFFT, TFFF
        # 1 zero, and 1 one flip: FTFT, FFTT, TTFF, TFTF
        # 2 zero flip: TTTT
        # # 2 one flip: FFFF
        p = sparsity
        # 1(no flip) + 2(1 zero flip) + 2(1 one flip) + 4(1zero and 1 one flip) + 1 (2 zero flip) + 1 (2 one flip)
        p_should = (
            p ** 2 * (1 - p) ** 2
            + 2 * p ** 3 * (1 - p) ** 1
            + 2 * p ** 1 * (1 - p) ** 3
            + 4 * p ** 2 * (1 - p) ** 2
            + p ** 4
            + (1 - p) ** 4
        )

        sim = np.sum(pattern == permutation)
        p_val = p_value_snapshot_same_sparsity(sim, sparsity, pattern)

        self.assertTrue(abs(p_val - p_should) < 0.002)


class TestClassSimilarityThreshold(TestCase):
    # note that the edge case pvalue = significance value is not reliably testable due to the approximation of the p value

    def test_when_sign_value_provided_is_lower_bounded_by_p_value_of_one_of_the_provided_similarity_values_should_return_corresponding_value_and_its_p_value(
        self,
    ):
        pattern = np.array(
            [True, False, False, True]
        )  # np.random.choice([True, False], p=[sparsity, 1-sparsity], size=size)
        size = pattern.size
        sparsity = 0.5

        # p value given pattern, sparsity - for two flips ~ similarity = size - nflips = 4 -2 = 2
        p_val_two_flip = 0.6875

        permutation = np.array([False, True, False, True])
        sim = np.sum(pattern == permutation)  # = 2
        crit = similarity_threshold(
            similarity=np.arange(0, size + 1),
            sparsity=sparsity,
            pattern=pattern,
            spike_count=int(pattern.size * 100 * sparsity),
            num_cycles=100,
            significance=p_val_two_flip + 0.01,
        )

        self.assertTrue(crit[0] == sim and abs(crit[1] - p_val_two_flip) < 0.003)

    def test_when_no_lower_bound_exists_on_sign_value_should_return_value_corresponding_to_no_critical_value_found(
        self,
    ):
        pattern = np.array(
            [True, False, False, True]
        )  # np.random.choice([True, False], p=[sparsity, 1-sparsity], size=size)
        size = pattern.size
        sparsity = 0.5

        permutation = np.array([False, True, False, True])
        # note that for similarity==esize (max): p(similarity) > 0.0 ie there is no similarity value for which p value is 0
        crit = similarity_threshold(
            similarity=np.arange(0, size + 1),
            sparsity=sparsity,
            pattern=pattern,
            spike_count=int(pattern.size * 100 * sparsity),
            num_cycles=100,
            significance=0.0,
        )

        self.assertEqual(crit, (None, None))


class TestClassFractionSignificantSnapshots(TestCase):
    def test_when_providing_basic_input_should_compute_pvalue(self):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array([[0.15, 0.012, 0.049], [0.1, 0.051, 0.032]])

        # that's the pattern
        pval = pvalue[0]
        sign_snaps_should = np.sum(pval <= significance) / pval.size
        sign_snaps_result, _ = attractor.fraction_significant_snapshots(
            pvalue, stimulus_pattern, pattern=pattern, significance=significance
        )

        self.assertEqual(sign_snaps_should, sign_snaps_result)


class TestClassFractionSignificantSnapshotsAcrossIntervals(TestCase):
    def test_when_first_trough_before_peak_should_compute_fraction_of_sign_snaps_for_intervals(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # t_peak < t_end && t_peak >= t_beg || t_snap_beg<= t_beg && t_snap_end >= t_end
        # it_0([9,19]): snaps 0,1 as peaks[0],peaks[1] in it_0
        # it_1([27,40]): snaps 3,4,5 as peaks[3], peaks[4], peaks[5] in it_1
        # it_2([51,53]): snaps 8 as peaks[8] int it_2
        interval = [(9.0, 19.0), (27.0, 40.0), (51.0, 53.0)]

        (
            sign_snaps_result,
            _,
        ) = attractor.fraction_significant_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            interval,
            significance=significance,
        )

        # that's the pattern
        pval = pvalue[0]
        # snaps contained in union of all intervals (note that fct enforces no overlap)
        idx = np.array([0, 1, 3, 4, 5, 8])
        pval_should = pval[idx]
        sign_snaps_should = np.sum(pval_should <= significance) / pval_should.size

        self.assertEqual(sign_snaps_should, sign_snaps_result)

    def test_when_first_trough_after_peak_should_compute_fraction_of_sign_snaps_for_intervals(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # peak first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array(
            [8.0, 12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3]
        )

        # t_peak < t_end && t_peak >= t_beg || t_snap_beg<= t_beg && t_snap_end >= t_end
        # it_0([9,19]): snaps 0,1 as peaks[0],peaks[1] in it_0
        # it_1([27,40]): snaps 3,4,5 as peaks[3], peaks[4], peaks[5] in it_1
        # it_2([51,53]): snaps 8 as peaks[8] int it_2
        interval = [(9.0, 19.0), (27.0, 40.0), (51.0, 53.0)]

        (
            sign_snaps_result,
            _,
        ) = attractor.fraction_significant_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            interval,
            significance=significance,
        )

        # that's the pattern
        pval = pvalue[0]
        # snaps contained in union of all intervals (note that fct enforces no overlap)
        idx = np.array([0, 1, 3, 4, 5, 8])
        pval_should = pval[idx]
        sign_snaps_should = np.sum(pval_should <= significance) / pval_should.size

        self.assertEqual(sign_snaps_should, sign_snaps_result)

    def test_when_interval_enclosed_by_snap_should_include_snap_in_computation(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # t_snap_beg<= t_beg && t_snap_end >= t_end
        interval = [(10.5, 12.0), (29.0, 30.0), (51.0, 53.0)]

        (
            sign_snaps_result,
            _,
        ) = attractor.fraction_significant_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            interval,
            significance=significance,
        )

        # that's the pattern
        pval = pvalue[0]
        # snaps contained in union of all intervals (note that fct enforces no overlap)
        idx = np.array([0, 3, 8])
        pval_should = pval[idx]
        sign_snaps_should = np.sum(pval_should <= significance) / pval_should.size

        self.assertEqual(sign_snaps_should, sign_snaps_result)

    def test_when_interval_starts_before_some_snap_and_ends_after_snap_peak_should_include_snap_in_computation(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # t_peak < t_end && t_peak >= t_beg
        interval = [(9.5, 12.4), (24.5, 29.7), (49.6, 52.1)]

        (
            sign_snaps_result,
            _,
        ) = attractor.fraction_significant_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            interval,
            significance=significance,
        )

        # that's the pattern
        pval = pvalue[0]
        # snaps contained in union of all intervals (note that fct enforces no overlap)
        idx = np.array([0, 3, 8])
        pval_should = pval[idx]
        sign_snaps_should = np.sum(pval_should <= significance) / pval_should.size

        self.assertEqual(sign_snaps_should, sign_snaps_result)

    def test_when_interval_starts_at_or_before_snap_peak_and_ends_after_snap_end_should_include_snap_in_computation(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # t_peak < t_end && t_peak >= t_beg
        interval = [(12.3, 15.1), (29.0, 31.0), (52.0, 55.0)]

        (
            sign_snaps_result,
            _,
        ) = attractor.fraction_significant_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            interval,
            significance=significance,
        )

        # that's the pattern
        pval = pvalue[0]
        # snaps contained in union of all intervals (note that fct enforces no overlap)
        idx = np.array([0, 3, 8])
        pval_should = pval[idx]
        sign_snaps_should = np.sum(pval_should <= significance) / pval_should.size

        self.assertEqual(sign_snaps_should, sign_snaps_result)

    def test_when_interval_starts_before_some_snap_and_ends_at_or_before_snap_peak_should_exclude_snap_in_computation(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # t_peak < t_end && t_peak >= t_beg
        interval = [(9.5, 12.3), (24.5, 29.5), (49.6, 52.0)]

        (
            sign_snaps_result,
            _,
        ) = attractor.fraction_significant_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            interval,
            significance=significance,
        )

        # that's the pattern
        # snaps contained in union of all intervals (note that fct enforces no overlap)
        sign_snaps_should = 0

        self.assertEqual(sign_snaps_should, sign_snaps_result)

    def test_when_interval_starts_after_snap_peak_and_ends_after_snap_end_should_exclude_snap_in_computation(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # t_peak < t_end && t_peak >= t_beg
        interval = [(12.4, 15.1), (29.7, 31.0), (52.1, 55.0)]

        (
            sign_snaps_result,
            _,
        ) = attractor.fraction_significant_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            interval,
            significance=significance,
        )

        sign_snaps_should = 0

        self.assertEqual(sign_snaps_should, sign_snaps_result)

    def test_when_interval_is_mix_between_enclosed_intervals_and_intervals_enclosing_snap_peaks_should_compute_fraction_of_sign_snaps_for_intervals(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # t_peak < t_end && t_peak >= t_beg || t_snap_beg<= t_beg && t_snap_end >= t_end
        # it_0([9,19]): snaps 0,1 as peaks[0],peaks[1] in it_0
        # it_1([27,28]): snaps 3 peaks[3] it_1
        # it_2([51,53]): snaps 8 as peaks[8] int it_2
        interval = [(9.0, 19.0), (27.0, 28.0), (51.0, 53.0)]

        (
            sign_snaps_result,
            _,
        ) = attractor.fraction_significant_snapshots_across_intervals(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            interval,
            significance=significance,
        )

        # that's the pattern
        pval = pvalue[0]
        # snaps contained in union of all intervals (note that fct enforces no overlap)
        idx = np.array([0, 1, 3, 8])
        pval_should = pval[idx]
        sign_snaps_should = np.sum(pval_should <= significance) / pval_should.size

        self.assertEqual(sign_snaps_should, sign_snaps_result)

    def test_when_one_interval_encloses_another_should_raise_value_error(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # enclosed
        interval = [(9.0, 19.0), (9.0, 19.0), (51.0, 53.0)]

        with self.assertRaises(ValueError):
            attractor.fraction_significant_snapshots_across_intervals(
                pvalue,
                stimulus_pattern,
                pattern,
                troughs,
                peaks,
                interval,
                significance=significance,
            )

    def test_when_one_interval_overlaps_with_another_should_raise_value_error(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # enclosed
        interval = [(9.0, 19.0), (15.0, 25.0), (51.0, 53.0)]

        with self.assertRaises(ValueError):
            attractor.fraction_significant_snapshots_across_intervals(
                pvalue,
                stimulus_pattern,
                pattern,
                troughs,
                peaks,
                interval,
                significance=significance,
            )


class TestClassFractionSignificantSnapshotsBlockedStimulus(TestCase):
    def test_basic_should_compute_fraction_of_sign_snaps_for_the_three_sections_before_at_after_stimulus_across_stimulus_onsets(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # t_peak < t_end && t_peak >= t_beg || t_snap_beg<= t_beg && t_snap_end >= t_end
        # snap indices for each of the 3 sections (before, at, after)
        # it_0([9,11]): (7,9),(9,11),(11,13)       -> (/,/,0)
        # it_1([27,29]): (25,27),(27,29),(29,31)   -> (3,3,3)
        # it_2([51,53]): (49,51), (51,53), (53,55) -> (/,8,/)
        stimulus_onset = np.array([9.0, 27.0, 51.0])
        stimulus_length = 2.0

        # snaps contained in union of all intervals (note that fct enforces no overlap)
        idx_before = np.array([3])
        idx_at = np.array([3, 8])
        idx_after = np.array([0, 3])

        (
            sign_snaps_result,
            _,
        ) = attractor.fraction_significant_snapshots_blocked_stimulus(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            stimulus_onset,
            stimulus_length,
            significance=significance,
        )

        # that's the pattern
        pval = pvalue[0]
        pval_should = pval[idx_before], pval[idx_at], pval[idx_after]
        sign_snaps_should = tuple(
            [np.sum(ps <= significance) / ps.size for ps in pval_should]
        )

        self.assertEqual(sign_snaps_should, sign_snaps_result)
        # fraction_significant_snapshots_blocked_stimulus


class TestClassFractionSignificantSnapshotsBlockedStimulusSlidingWindow(TestCase):
    def test_basic_should_compute_fraction_of_sign_snaps_for_sliding_window(
        self,
    ):
        significance = 0.05
        # (N, pattern_length) where N is # of patterns
        pattern = np.array([[True, False, True], [False, True, False]])
        stimulus_pattern = pattern[0]
        # (N, C) where N is # patterns and C is # snaps - assume 3 snaps
        pvalue = np.array(
            [
                [0.15, 0.012, 0.049, 0.15, 0.012, 0.049, 0.15, 0.012, 0.049],
                [0.1, 0.051, 0.032, 0.1, 0.051, 0.032, 0.1, 0.051, 0.032],
            ]
        )

        # trough first
        troughs = np.array(
            [10.0, 15.0, 21.2, 25.0, 30.97, 37.0, 41.0, 46.0, 49.7, 54.0]
        )
        peaks = np.array([12.3, 17.0, 23.0, 29.6, 34.2, 39.0, 43.0, 47.8, 52.0, 56.3])

        # t_peak < t_end && t_peak >= t_beg || t_snap_beg<= t_beg && t_snap_end >= t_end
        # it_0([9,11],  [21,23], [33,35], [45,47]):  -> (/, 2, 4, /)
        # it_1([11,13], [23,25], [35,37], [47,49]):  -> (0, 2, 4, 7)
        # it_2([13,15], [25,27], [37,39], [49,51]):  -> (0, 3, 5, /)
        # it_3([15,17], [27,29], [39,41], [51,53]):  -> (1, 3, 5, 8)
        # it_4([17,19], [29,31], [41,43], [53,55]):  -> (1, 3, 6, /)
        # it_5([19,21], [31,33], [43,45], [55,57]):  -> (1, 4, 6, /)
        #    note  for it_5: while 55,57 encloses peak 56 there is no snapshot enclosing the peak 56 so it does not contribute

        stimulus_onset = 10.0
        inter_onset_interval = 12.0
        # t_end >= last peak in snapshot + inter_onset_interval
        t_end = 64.0
        window_length = 2.0
        window_step = 2.0
        # stimulus_length = 2.0

        # snaps contained in union of all intervals (note that fct enforces no overlap)
        idx = [
            np.array([2, 4]),
            np.array([0, 2, 4, 7]),
            np.array([0, 3, 5]),
            np.array([1, 3, 5, 8]),
            np.array([1, 3, 6]),
            np.array([1, 4, 6]),
        ]

        (
            sign_snaps_result,
            _,
        ) = attractor.fraction_significant_snapshots_blocked_stimulus_sliding_window(
            pvalue,
            stimulus_pattern,
            pattern,
            troughs,
            peaks,
            stimulus_onset,
            t_end,
            inter_onset_interval,
            window_length,
            window_step,
            significance=significance,
        )

        # using pattern 0
        pval = pvalue[0]
        pval_should = [pval[i] for i in idx]
        sign_snaps_should = tuple(
            [np.sum(ps <= significance) / ps.size for ps in pval_should]
        )

        # raise ValueError(np.array(sign_snaps_should) == np.array(sign_snaps_result), sign_snaps_should, sign_snaps_result)

        self.assertEqual(sign_snaps_should, sign_snaps_result)
        # fraction_significant_snapshots_blocked_stimulus


class TestClassSeparatePresentationCycles(TestCase):
    def test_basic_should_separate_presentation_cycles(
        self,
    ):
        # 2 cases: i) distance to trough is smaller ii) if distance equal (abs(peak-so) == abs(trough-so)), then two cases: a) peak < so < trough and b) trough < so < peak
        # -> trough cycle cases: i) and iia)
        # 1.0 -> peak, 7.5 -> peaks (edge case iib)), 12.5 -> troughs (edge case iia), 16-> trough, 37-> trough, 52 -> peak]
        #  (no markup means case i))

        stimulus_onset = np.array([1.0, 7.5, 12.5, 16, 37, 52])

        peaks = np.arange(0, 100, 10)
        troughs = np.arange(5, 105, 10)

        trough_cycle, peak_cycle = attractor.separate_presentation_cycles(
            troughs, peaks, stimulus_onset
        )

        should_trough_cycle = [2, 3, 4]
        should_peak_cycle = [0, 1, 5]

        self.assertEqual(trough_cycle.tolist(), should_trough_cycle)
        self.assertEqual(peak_cycle.tolist(), should_peak_cycle)
