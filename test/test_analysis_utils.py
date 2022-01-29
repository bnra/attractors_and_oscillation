import numpy as np

from test.utils import TestCase
import analysis_utils


class TestClassFctRectifySubsequentFallingRisingEdgeWithoutPlats(TestCase):
    def test_when_providing_indices_of_extrema_where_falling_edge_idx_is_in_mins_and_rising_edge_idx_in_max_and_vice_versa_and_whose_corresponding_values_in_the_vector_are_the_same_should_remove_these_indices(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 5, 5, 7, 8, 7, 6, 6, 4])
        mins_idx = np.array([6, 10])
        maxs_idx = np.array([5, 8, 11])
        mins_should = np.array([])
        maxs_should = np.array([8])
        (
            mins_is,
            maxs_is,
        ) = analysis_utils.rectify_subsequent_falling_rising_edge_without_plats(
            x, mins_idx, maxs_idx
        )
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )

    def test_when_providing_indices_of_extrema_where_falling_edge_idx_is_in_mins_and_rising_edge_idx_in_max_and_vice_versa_and_whose_corresponding_values_in_the_vector_are_not_equal_should_not_remove_these_indices(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 3, 6, 3, 2, 1, 0, -1, -2])
        mins_idx = np.array([5])
        maxs_idx = np.array([4, 6])
        mins_should = np.array([5])
        maxs_should = np.array([4, 6])

        (
            mins_is,
            maxs_is,
        ) = analysis_utils.rectify_subsequent_falling_rising_edge_without_plats(
            x, mins_idx, maxs_idx
        )
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )

    def test_when_providing_indices_where_falling_and_rising_edge_without_plateau_are_both_peaks_or_both_troughs_should_remove_the_falling_edges(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3])
        mins_idx = np.array([8, 9])
        maxs_idx = np.array([4, 5])
        mins_should = np.array([8])
        maxs_should = np.array([4])

        (
            mins_is,
            maxs_is,
        ) = analysis_utils.rectify_subsequent_falling_rising_edge_without_plats(
            x, mins_idx, maxs_idx
        )
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )


class TestClassFctRectifyPlats(TestCase):
    def test_when_given_falling_and_rising_edge_both_peaks_or_both_troughs_separated_by_plateaus_should_replace_these_indices_with_the_middle_index(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 1, 1, 1, 2, 3])
        mins_idx = np.array([9, 12])
        maxs_idx = np.array([4, 6])
        mins_mask = np.zeros_like(x)
        mins_mask[mins_idx] = True
        maxs_mask = np.zeros_like(x)
        maxs_mask[maxs_idx] = True

        plateaus = [
            (4, 6),
            (9, 12),
        ]  # note this is computed from plateaus np.array([5,10,11]) by from_plateaus_compute_falling_and_rising_edge

        mins_should_idx = np.array([10])
        maxs_should_idx = np.array([5])
        mins_should_mask = np.zeros_like(x)
        mins_should_mask[mins_should_idx] = True
        maxs_should_mask = np.zeros_like(x)
        maxs_should_mask[maxs_should_idx] = True

        mins_is, maxs_is = analysis_utils.rectify_plats(plateaus, mins_mask, maxs_mask)

        self.assertTrue(
            np.all(mins_should_mask == mins_is) and np.all(maxs_is == maxs_should_mask)
        )

    def test_when_given_falling_edge_of_peak_and_rising_edge_of_trough_or_vice_versa_separated_by_plateaus_should_remove_these_indices(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 4, 4, 5, 6, 7, 6, 5, 4, 4, 4, 4, 3, 2])
        mins_idx = np.array([6, 12])
        maxs_idx = np.array([4, 9, 15])

        mins_mask = np.zeros_like(x)
        mins_mask[mins_idx] = True
        maxs_mask = np.zeros_like(x)
        maxs_mask[maxs_idx] = True

        plateaus = [
            (4, 6),
            (12, 15),
        ]  # note this is computed from plateaus np.array([5,10,11]) by from_plateaus_compute_falling_and_rising_edge

        mins_should_idx = np.array([])
        maxs_should_idx = np.array([9])
        mins_should_mask = np.zeros_like(x)  # note mins_should_idx is empty
        maxs_should_mask = np.zeros_like(x)
        maxs_should_mask[maxs_should_idx] = True

        mins_is, maxs_is = analysis_utils.rectify_plats(plateaus, mins_mask, maxs_mask)

        self.assertTrue(
            np.all(mins_should_mask == mins_is) and np.all(maxs_is == maxs_should_mask)
        )


class TestClassFctFromPlateausComputeFallingAndRisingEdge(TestCase):
    def test_when_given_falling_and_rising_edge_separated_by_plateaus_should_extract_rising_and_falling_edges(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 1, 1, 1, 2, 3])
        plateau_idx = np.array([5, 10, 11])

        result = analysis_utils.from_plateaus_compute_falling_and_rising_edges(
            x, plateau_idx
        )
        self.assertEqual(result, [(4, 6), (9, 12)])

    def test_when_given_falling_and_rising_edge_separated_by_plateaus_should_not_extract_edge_case_at_beginning_or_start(
        self,
    ):
        x = np.array([0, 0, 1, 2, 2, 2])
        plateau_idx = np.array([0, 4, 5])

        result = analysis_utils.from_plateaus_compute_falling_and_rising_edges(
            x, plateau_idx
        )
        self.assertEqual(result, [])


class TestClassFctDetectPeaks(TestCase):
    def test_when_providing_vector_with_extrema_where_falling_edge_idx_is_in_mins_and_rising_edge_idx_in_max_and_vice_versa_and_whose_corresponding_values_in_the_vector_are_the_same_should_not_detect_these_indices(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 5, 5, 7, 8, 7, 6, 6, 4])
        mins_should = np.array([])
        maxs_should = np.array([8])

        mins_is, maxs_is = analysis_utils.detect_peaks(x)
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )

    def test_when_providing_vector_with_extrema_where_falling_edge_idx_is_in_mins_and_rising_edge_idx_in_max_and_vice_versa_and_whose_corresponding_values_in_the_vector_are_not_equal_should_not_remove_these_indices(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 3, 6, 3, 2, 1, 0, -1, -2])
        mins_should = np.array([5])
        maxs_should = np.array([4, 6])

        mins_is, maxs_is = analysis_utils.detect_peaks(x)
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )

    def test_when_providing_vector_where_falling_and_rising_edge_without_plateau_are_both_peaks_or_both_troughs_should_remove_the_falling_edges(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3])
        mins_should = np.array([8])
        maxs_should = np.array([4])

        mins_is, maxs_is = analysis_utils.detect_peaks(x)
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )

    def test_when_given_vector_with_falling_and_rising_edge_separated_by_plateaus_should_return_middle_index(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 1, 1, 1, 2, 3])
        mins_should = np.array([10])
        maxs_should = np.array([5])

        mins_is, maxs_is = analysis_utils.detect_peaks(x)
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )

    def test_when_given_vector_with_falling_and_rising_edge_separated_by_plateaus_should_not_extract_edge_case_at_beginning_or_start_of_vector(
        self,
    ):
        x = np.array([0, 0, 1, 2, 2, 2])
        mins_should = np.array([])
        maxs_should = np.array([])

        mins_is, maxs_is = analysis_utils.detect_peaks(x)
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )

    def test_when_given_vector_with_falling_edge_of_peak_and_rising_edge_of_trough_or_vice_versa_separated_by_plateaus_should_remove_these_indices(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 4, 4, 5, 6, 7, 6, 5, 4, 4, 4, 4, 3, 2])

        mins_should = np.array([])
        maxs_should = np.array([9])

        mins_is, maxs_is = analysis_utils.detect_peaks(x)
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )

    def test_when_passing_vector_with_many_peaks_and_troughs_should_detect_all_peaks_and_troughs(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 3, 2, 3, 2, 3, 3, 2, 3, 2, 1, 0, 1, 2, 3])

        mins_should = np.array([6, 8, 11, 15])
        maxs_should = np.array([4, 7, 9, 12])

        mins_is, maxs_is = analysis_utils.detect_peaks(x)
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )


class TestClassFctDetectSymmetricPeaks(TestCase):
    def test_when_peaks_and_troughs_below_delta_should_not_detect_them(self):
        x = np.array([0, 1, 2, 3, 4, 3, 2, 3, 2, 3, 3, 2, 3, 2, 1, 0, 1, 2, 3])

        mins_should = np.array([15])
        maxs_should = np.array([4])

        mins_is, maxs_is = analysis_utils.detect_symmetric_peaks(x, delta=3.0)
        # raise ValueError(mins_should, mins_is, maxs_is, maxs_should)
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )

    def test_when_delta_peak_followed_by_subthreshold_trough_followed_by_delta_peak_should_add_all_three_extrema(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 3, 2, 3, 4, 5, 6, 7, 6, 5, 4])

        mins_should = np.array([6])
        maxs_should = np.array([4, 11])

        mins_is, maxs_is = analysis_utils.detect_symmetric_peaks(x, delta=3.0)
        # raise ValueError(mins_should, mins_is, maxs_is, maxs_should)
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )

    def test_when_delta_peak_followed_by_subthreshold_trough_followed_by_subtreshold_peak_followed_by_delta_trough_should_add_delta_peak_and_delta_trough_only_to_mins_and_maxs_resp(
        self,
    ):
        x = np.array([0, 1, 2, 3, 4, 3, 2, 3, 2, 1, 0, 1, 2, 3])

        mins_should = np.array([10])
        maxs_should = np.array([4])

        mins_is, maxs_is = analysis_utils.detect_symmetric_peaks(x, delta=3.0)
        # raise ValueError(mins_should, mins_is, maxs_is, maxs_should)
        self.assertTrue(
            np.all(mins_should == mins_is) and np.all(maxs_is == maxs_should)
        )
