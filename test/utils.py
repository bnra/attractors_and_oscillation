import os
import shutil
from typing import Iterable, Tuple
import unittest


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
        self.tmp_base_name = ".tmp"
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