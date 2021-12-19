import numpy as np
from numpy.lib.arraysetops import unique
import tables

from test.utils import SpeedTest
from persistence import FileMap, Writer, Node
from utils import unique_idx, clean_brian2_quantity
from brian2 import PoissonGroup, SpikeMonitor, run, ms, kHz, Network

class SaveFileMap(SpeedTest):
    
    trials = 3
    iterations = 5

    def setUp(self):
        self.data = np.arange(4e8, dtype=float).reshape(4000, 100000)

    def run(self):
        with FileMap(path="file.h5",mode="write") as f:
            f["bla"] = self.data

class SpikeMonitorGetStates(SpeedTest):
    trials = 3
    iterations = 10
    def setUp(self):
        self._pop = PoissonGroup(5000, rates=10000 * kHz)
        self._spike = SpikeMonitor(self._pop)
        self._network = Network()
        self._network.add(self._pop)
        self._network.add(self._spike)
        self._network.run(1000 * ms)

    def run(self):
        states = self._spike.get_states()
        idx, vals = states["i"], states["t"]
        ids, indices = unique_idx(idx)
        vals, unit = clean_brian2_quantity(vals)

        data = {}


        data["spike_train"] = {str(i): vals[idx] for i, idx in zip(ids, indices)}

        data["meta"] = {"spike_train": str(unit)}

class SpikeMonitorAllValues(SpeedTest):
    trials = 3
    iterations = 10
    def setUp(self):
        self._pop = PoissonGroup(5000, rates=10000 * kHz)
        self._spike = SpikeMonitor(self._pop)
        self._network = Network()
        self._network.add(self._pop)
        self._network.add(self._spike)
        self._network.run(1000 * ms)

    def run(self):
        recs = self._spike.all_values()["t"]
        val = list(recs.values())[0]
        unit = val.get_best_unit()
        data = {}        
        data["spike_train"] = {
            str(k): np.asarray(v / unit) for k, v in recs.items()
        }

class NumpyGetUniqueIndices(SpeedTest):
    
    trials = 3
    iterations = 5

    def setUp(self):
        self.data = np.tile(np.arange(10000), 4000).reshape(4000, 10000)

    def run(self):
        vals = np.unique(self.data)
        idx = np.array([np.where(self.data == e) for e in vals])

class NumpyGetUnique(SpeedTest):
    
    trials = 3
    iterations = 5

    def setUp(self):
        self.data = np.tile(np.arange(10000), 4000).reshape(4000, 10000)

    def run(self):
        vals = np.unique(self.data)

class NumpyUniqueIdx(SpeedTest):
    
    trials = 3
    iterations = 5

    def setUp(self):
        self.data = np.tile(np.arange(10000), 4000).reshape(4000, 10000)

    def run(self):
        vals, idx = unique_idx(self.data.ravel())


class BasicWriter(SpeedTest):
    
    trials = 3
    iterations = 5

    def setUp(self):
        self.file = tables.File("run.h5", mode="w")
        self.writer = Writer(self.file, object_path="/")
        tmp = self.writer
        for i in range(10):
            tmp[f"l_{i}"] = Node()
            tmp = tmp[f"l_{i}"]


    def run(self):
        self.writer["l_0"]["l_1"]["l_2"]["l_3"]["l_4"]["l_5"]["l_6"]["l_7"]["l_8"]["l_9"]

    def tearDown(self):
        self.file.close()
        



class SaveArrayFile(SpeedTest):
    
    trials = 3
    iterations = 5

    def setUp(self):
        self.data = np.arange(int(4e8)).reshape(4000, 100000)

    def run(self):
        f = tables.File("file.h5", mode="w")
        f.create_array("/" ,"bla", obj=self.data)
        f.close()


class SaveCarrayFile(SpeedTest):
    
    trials = 3
    iterations = 5

    def setUp(self):
        self.data = np.arange(int(4e8)).reshape(4000, 100000)

    def run(self):
        f = tables.File("file.h5", mode="w")
        f.create_carray("/" ,"bla", obj=self.data)
        f.close()


