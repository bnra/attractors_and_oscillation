from distribution import draw_bernoulli, draw_normal, draw_uniform, draw_uniformly_random_from_values
from test.utils import TestCase
import numpy as np

class TestFctDrawUniform(TestCase):

    def test_when_called_should_draw_randomly_from_uniform_distribution(self):
        n = 3000
        bin_num = 10
        vals = draw_uniform(a=0., b=1., size=n)
        bins = [0 for _ in range(bin_num)]
        for v in vals:
            if v == 1.0:
                bin[-1] += 1
            else:
                # treat 0.0-0.099.., 0.1-0.199.., ... 0.9-0.999.. 
                bins[int((v * 10)//1.0)] += 1

        uniform_num = n/bin_num

        #raise ValueError(f"{[abs(b - uniform_num) / uniform_num for b in bins]}")
        self.assertTrue(all([abs(b - uniform_num) / uniform_num < 0.2 for b in bins]))

class TestFctDrawNormal(TestCase):    
    def test_when_called_should_draw_randomly_from_normal_distribution(self):
        n = 3000
        mu = 0.0
        sigma = 1.0
        vals = draw_normal(mu=mu, sigma=sigma, size=n)
        mean = np.mean(vals)
        std = np.std(vals)
        self.assertTrue(abs(mu-mean) < 0.1 and (sigma - std) / sigma < 0.1)

class TestFctDrawBernoulli(TestCase):    
    def test_when_called_should_draw_randomly_from_bernoulli_distribution(self):
        n = 1000
        p = 0.3
        vals = draw_bernoulli(p=p, size=n)
        self.assertTrue(abs(n*p - np.sum(vals))/ (n * p) < 0.1)


class TestFctDrawUniformelyRandomFromValues(TestCase):
    def test_when_called_should_draw_uniformly_random_from_passed_values(self):
        n = 3000
        val_range = 10
        vals = draw_uniformly_random_from_values(np.arange(val_range), size=n)
        bins = [0 for _ in range(val_range)]
        for v in vals:
            bins[v] += 1
        uniform_num = n/val_range
        
        #raise ValueError(f"{[abs(b - uniform_num) / uniform_num for b in bins]}")
        self.assertTrue(all([abs(b - uniform_num) / uniform_num < 0.2 for b in bins]))

