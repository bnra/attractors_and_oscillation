from test.utils import TestCase
from utils import TestEnv

import numpy as np
import multiprocessing
import os
import sys
import itertools
from typing import List

import mp


class TestClassCaptureStandardStreams(TestCase):
    def test_when_stderr_and_stdout_captured_should_return_corresponding_str(self):
        out = "hi"
        err = "bye"
        capture_streams = mp.CaptureStandardStreams(stdout=True, stderr=True)
        with capture_streams:
            print(out)
            print(err, file=sys.stderr)

        self.assertEqual(out + "\n", capture_streams.stdout)
        self.assertEqual(err + "\n", capture_streams.stderr)


class TestClassMultiPipeCommunication(TestCase):
    def test_when_communicating_within_a_single_process_should_receive_msg(self):
        p_com, c_com = mp.MultiPipeCommunication(stream_names=["stdout"])
        msg = "hi"
        c_com.send(msg, stream=c_com.stdout)
        msg2 = "bye"
        c_com.send(msg2, stream=c_com.stdout)

        result = p_com.recv(stream=p_com.stdout)
        self.assertEqual([r for r, t in result], [msg, msg2])

    def test_when_communicating_across_processes_should_receive_msg(self):
        p_com, c_com = mp.MultiPipeCommunication(stream_names=["stdout"])
        msg = "hi"
        msg2 = "bye"

        def msging(c_com, msg, msg2):
            msg = msg
            c_com.send(msg, stream=c_com.stdout)
            msg2 = msg2
            c_com.send(msg2, stream=c_com.stdout)

        P = multiprocessing.Process(target=msging, args=(c_com, msg, msg2))

        P.start()

        P.join()

        result = p_com.recv(stream=p_com.stdout)
        self.assertEqual([r for r, t in result], [msg, msg2])

    def test_when_communicating_with_multiple_streams_should_handle_separately(self):
        p_com, c_com = mp.MultiPipeCommunication(stream_names=["stdout", "stderr"])
        msg = "hi"
        c_com.send(msg, stream=c_com.stdout)
        msg2 = "bye"
        c_com.send(msg2, stream=c_com.stdout)
        msg3 = "bla"
        c_com.send(msg3, stream=c_com.stderr)
        msg4 = "blu"
        c_com.send(msg4, stream=c_com.stderr)

        out = p_com.recv(stream=p_com.stdout)
        err = p_com.recv(stream=p_com.stderr)

        self.assertEqual([o for o, t in out], [msg, msg2])
        self.assertEqual([e for e, t in err], [msg3, msg4])

    def test_when_printing_should_return_no_empty_lines(self):
        p_com, c_com = mp.MultiPipeCommunication(stream_names=["stdout"])
        # print(s) ~ write(s+"\n")
        corem = "hi"
        corem2 = "bye"
        msg = f"{corem}\n"
        c_com.send(msg, stream=c_com.stdout)
        msg2 = f"{corem2}\n"
        c_com.send(msg2, stream=c_com.stdout)
        out = p_com.recvlines(stream=p_com.stdout, keep_empty=True)

        self.assertEqual([o for o, t in out], [[corem], [corem2]])

    def test_when_printing_multiple_new_lines_should_return_same_amount_of_empty_msgs(
        self,
    ):
        count = 3
        p_com, c_com = mp.MultiPipeCommunication(stream_names=["stdout"])
        # print(s) ~ write(s+"\n")
        corem = "hi"
        corem2 = "bye"
        msg = f"{corem}\n" + "\n" * count
        c_com.send(msg, stream=c_com.stdout)
        msg2 = f"{corem2}\n" + "\n" * count
        c_com.send(msg2, stream=c_com.stdout)
        out = p_com.recvlines(stream=p_com.stdout, keep_empty=True)

        self.assertEqual(
            [o for o, t in out],
            [["hi"] + ["" for _ in range(count)], ["bye"] + ["" for _ in range(count)]],
        )


class TestClassProcessExperiment(TestCase):
    # use logging instead of setting stdout to temporary file

    def test_when_no_com_passed_should_not_raise_no_such_attribute_error(self):
        b = [2, 3]
        data = [0, 1]
        acc = [1, 2]

        def fmap(path, b, data, acc):

            x = np.math.factorial(b * data + acc)
            print(f"{b,data,acc} -> {x}")

            return x

        with TestEnv() as env:
            os.makedirs(os.path.join(env.tmp_dir, "bla"))
            # not that no communication object parameter 'com' is passed
            proc = mp.ProcessExperiment(
                0,
                1,
                {"path": "bla/a.h5", "b": b, "data": data, "acc": acc},
                fmap,
            )

            proc.start()

            proc.join(3)

            alive = proc.is_alive()

            if alive:
                proc.terminate()
                proc.close()

            self.assertTrue(not alive)


class TestFctInParallel(TestCase):
    def test_when_writing_files_in_target_function_should_write_all_files(self):
        base_path = "base_path"
        fnames = [f"file_{i}.h5" for i in range(12)]

        def f(path: str, file_idx: int):
            # file_idx is needed to get file path in file_name_generator as
            # for each instance the file_name_generator is passed one instantiation of the cartesian product
            # of the parameters as k,v pairs ie (k1,v1),(k2,v2),(k3,v3) for 3 parameters
            with open(path, "w") as f:
                f.write("")

        with TestEnv() as env:
            os.makedirs(os.path.join(env.tmp_dir, base_path))
            pool = mp.Pool(
                base_path,
                {"file_idx": list(range(len(fnames)))},
                f,
                {},
                file_name_generator=lambda idx: fnames[idx[0][1]],
            )
            stdouts, stderrs = pool.run()

            result = os.listdir(base_path)

            self.assertTrue(
                len(fnames) == len(result) and all([f in result for f in fnames])
            )

    def test_when_distributing_range_across_processes_and_printing_result_should_return_all_results(
        self,
    ):

        b = [2, 3]
        data = list(range(10))
        acc = [1, 2]

        def fmap(path, b, data, acc):

            x = np.math.factorial(b * data + acc)
            print(f"{b,data,acc} -> {x}")

            return x

        with TestEnv() as env:
            os.makedirs(os.path.join(env.tmp_dir, "bla"))

            pool = mp.Pool(
                "bla",
                {"b": b, "data": data, "acc": acc},
                fmap,
                {},
            )
            stdouts, stderrs = pool.run()

            result = [
                s for sl in stdouts.values() for so, _ in sl for s in so if len(s) > 0
            ]

        should = [
            f"{be,de,ae} -> {np.math.factorial(be * de + ae)}"
            for be, de, ae in itertools.product(b, data, acc)
        ]

        self.assertTrue(
            all([s in result for s in should]) and len(should) == len(result)
        )
