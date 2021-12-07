
import numpy as np

from test.utils import TestCase
from connectivity import all2all, bernoulli

class TestFctAll2All(TestCase):
    def test_when_called_should_return_cartesian_product(self):
        should_source = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4,4, 4, 4]),
        should_dest = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1,2, 3, 4])
        is_source, is_dest = all2all(range(5), range(5))
        self.assertTrue(np.all(is_source == should_source))
        self.assertTrue(np.all(is_dest == should_dest))


class TestFctBernoulli(TestCase):
    def test_when_called_should_should_connect_each_source_to_each_dest_index_with_prob_p(self):
        source = np.arange(1000)
        dest = np.arange(1000)
        p = 0.5
        is_source, is_dest = bernoulli(source, dest, p)
        

        source_counts = np.unique(is_source, return_counts=True)[1]
        dest_counts = np.unique(is_dest, return_counts=True)[1]

        source_diff = abs(source_counts - p * dest.size)/(p*dest.size) 
        dest_diff = abs(dest_counts - p * source.size)/(p*source.size)

        #raise ValueError(f"source: {source_counts}, dest: {dest_counts}")
        #raise ValueError(f"source: {np.max(source_diff)}, dest: {np.max(dest_diff)}")
        self.assertTrue(source.size == dest_counts.size and np.all(source_diff < 0.15))
        self.assertTrue(dest.size == source_counts.size  and np.all(dest_diff < 0.15))

    def test_when_called_with_differing_pop_sizes_should_not_through_exception(self):
        source = np.arange(3)
        dest = np.arange(13)
        p = 0.1
        is_source, is_dest = bernoulli(source, dest, p)
        
        self.assertTrue(True)
