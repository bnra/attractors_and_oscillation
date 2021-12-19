from abc import ABCMeta, abstractmethod
import os
import shutil
from typing import Iterable, Tuple
import unittest
from tqdm import tqdm
import time
from typing import Callable, Tuple, Dict, Any, List
import sys


class TestCase(unittest.TestCase):
    """
    Test Case class that wraps :class:`unittest.TestCase` and provides some additional setup and teardown
    If you want to provide additional setup and teardown for a test case then add members _setUp and _tearDown

    .. testsetup::

        from test.utils import TestCase

    .. testcode::

        class TestMyCase(TestCase):
            def _setUp(self):
                print("setting up.")
            def _tearDown(self):
                print("tearing down.")
        t = TestMyCase()
        t.setUp()

        print("do stuff.")

        t.tearDown()

    .. testoutput::

        setting up.
        do stuff.
        tearing down.
        
    Calls t.setUp() which in turn calls t._setUp()
    """

    def __init__(self, *args, **kwargs):
        self.initial_dir = os.path.abspath(".")
        self.tmp_base_name = ".tmp_test"
        self.tmp_dir = os.path.abspath(self.tmp_base_name)
        super().__init__(*args, **kwargs)

    def setUp(self):
        os.makedirs(self.tmp_dir)
        os.chdir(self.tmp_dir)
        if hasattr(self, "_setUp"):
            self._setUp()

    def tearDown(self):
        if hasattr(self, "_tearDown"):
            self._tearDown()
        os.chdir(self.initial_dir)
        shutil.rmtree(self.tmp_dir)

            
def compare_iterables(should_dict:Iterable, is_dict:Iterable)->Tuple[Iterable,Iterable]:
    should_diff = [k for k in should_dict if k not in is_dict]
    is_diff = [k for k in is_dict if k not in should_dict]
    return should_diff, is_diff



class StdPipeSilencer(object):
    
    class DummyFile(object):
        def write(self, s): 
            pass
        def flush(self):
            pass

    def __init__(self, silence_out=True, silence_error=False):
        
        self.silence_out = silence_out
        self.silence_error = silence_error

        # backup writes to out and err from before entering context
        self.out_back = None
        self.error_back = None

    def __enter__(self):
        
        if self.silence_out:
            self.out_back = sys.stdout
            sys.stdout = StdPipeSilencer.DummyFile()
        
        if self.silence_error:
            self.out_error = sys.stderr
            sys.stderr = StdPipeSilencer.DummyFile() 

    def __exit__(self, exc_type, exc_value, tb):

        if self.silence_out:
            sys.stdout = self.out_back
            self.out_back = None
        
        if self.silence_error:
            sys.stderr = self.error_back
            self.error_back = None

        if exc_type != None:
            raise exc_type(exc_value, tb)


class SpeedTest(metaclass=ABCMeta):
    """
    abstract base class for speed tests
    subclass and 
    - (opt.) add setup code in def setUp()
    - (opt.) add teardown code in tearDown()
    - add test code in the sub classes run()
    - (opt., recommended) set # iterations (iterations) and # trials (trials) as class variables of subclass 
    """
    def __init__(self):
        self.tmp_base_name = ".tmp_speed_test"

        self.initial_dir = os.path.abspath(".")
        self.tmp_dir = os.path.abspath(self.tmp_base_name)

    def __enter__(self):
        os.makedirs(self.tmp_dir)
        os.chdir(self.tmp_dir)
        if hasattr(self, "setUp"):
            self.setUp()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if hasattr(self, "tearDown"):
            self.tearDown()
        os.chdir(self.initial_dir)
        shutil.rmtree(self.tmp_dir)

        if exc_type != None:
            raise exc_type(exc_value, tb)

    @abstractmethod
    def run():
        pass



class SpeedTester:
    """
    speed test a function - to time arbitrary code simply wrap in function
    """

    def __init__(
        self,
        test: SpeedTest,
        reduce: Callable[[List[float]], float] = lambda x: sum(x) / len(x),
    ):
        """
        :param test: test object used in testing
        :param reduce: function used for aggregation of results (eg. averaging) of time [ns]
        """
        self.test = test
        self.reduce = reduce
        self._statistics = None

    @property
    def statistics(self):
        if not self._statistics:
            raise Exception("Call only after run() called at least once.")
        return self._statistics

    def run(self, trials=7, iterations=1000) -> List[float]:
        """ "
        run the speedtest

        :param trials: number of trials that are run
        :param iterations: number of iterations run per trial - iterations are reduced to single scalar
                            measure by reduce parameter in :meth:`SpeedTester.__init__()`
        :return: set of single value statistics of each of the trials 
        """
        statistics = []



        for _ in tqdm(range(trials)):
            trial = []
            for _ in tqdm(range(iterations), leave=False):
                with StdPipeSilencer():
                    with self.test() as test:
                        start = time.time_ns()
                        test.run()
                        end = time.time_ns()
                    trial.append((end - start))

            statistics.append(self.reduce(trial))

        self._statistics = statistics

        return statistics

