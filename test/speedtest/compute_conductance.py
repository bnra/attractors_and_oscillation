import numpy as np

from test.utils import SpeedTest
from attractor import compute_conductance_scaling
from test.test_attractor import TestClassFctComputeConductanceScaling

class NaiveConductance(SpeedTest):
    
    trials = 3
    iterations = 5

    
    def setUp(self):
        size = 100
        p = 100

        self.sparsity = 0.1

        self.patterns = np.random.choice([True, False], p=[self.sparsity, 1.0 - self.sparsity], size=p*size).reshape(p, size)


        self.g_ee = 5e-8
        self.c_ee = 1

    def run(self):
        g = TestClassFctComputeConductanceScaling._naive_compute_conductance(self.patterns, self.sparsity, self.g_ee, self.c_ee)





    


class FastConductance(SpeedTest):

    trials = 3
    iterations = 5
    
    def setUp(self):
        size = 100
        p = 100

        self.sparsity = 0.1

        self.patterns = np.random.choice([True, False], p=[self.sparsity, 1.0 - self.sparsity], size=p*size).reshape(p, size)


        self.g_ee = 5e-8
        self.c_ee = 1


    def run(self):
        s = compute_conductance_scaling(self.patterns, self.sparsity)
        g = self.g_ee/self.c_ee * s
