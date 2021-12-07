import numpy as np
import tables

from test.utils import SpeedTest
from persistence import FileMap, Writer, Node


class SaveFileMap(SpeedTest):
    
    trials = 3
    iterations = 5

    def setUp(self):
        self.data = np.arange(4e8, dtype=float).reshape(4000, 100000)

    def run(self):
        with FileMap(path="file.h5",mode="write") as f:
            f["bla"] = self.data


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


