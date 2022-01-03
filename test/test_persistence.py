import tables

import numpy as np
import os


from test.utils import TestCase
from persistence import Reader, Writer, FileMap, Node, get_nodes, opath


class TestFctGetNodes(TestCase):
    def _setUp(self):
        path = "file.h5"
        file = tables.File(path, mode="w")
        file.create_group("/", "mydata")
        file.create_group("/mydata", "run_x")
        file.create_array("/mydata/run_x", "spikes", obj=np.arange(100))
        file.create_array("/", "array", obj=np.arange(100))
        file.close()

    def test_when_called_at_root_should_return_leaves_and_nodes(self):
        path = "file.h5"
        file = tables.File(path, mode="r+")
        nodes, leaves = get_nodes(file, "/")
        self.assertEqual(([*nodes.keys()], [*leaves.keys()]), (["mydata"], ["array"]))
        file.close()


class TestClassWriter(TestCase):
    def _setUp(self):
        path = "file.h5"
        file = tables.File(path, mode="w")
        file.create_group("/", "mydata")
        file.create_group("/mydata", "run_x")
        file.create_array("/mydata/run_x", "spikes", obj=np.arange(100))
        file.close()

    def test_when_assigning_to_writer_should_write_to_file(self):
        path = "file.h5"
        file = tables.File(path, mode="w")
        file.close()
        file = tables.File(path, mode="r+")

        w = Writer(file, "/")

        w["mydata"] = Node()
        md = w["mydata"]
        md["run_x"] = Node()
        m = md["run_x"]
        m["spikes"] = np.arange(200)
        w["array"] = np.arange(100)

        file.close()

        file = tables.File(path, mode="r")
        nodes, leaves = get_nodes(file, "/")
        _, second_array = get_nodes(file, "/mydata/run_x")

        self.assertEqual(["mydata"], list(nodes.keys()))
        self.assertEqual(["array"], list(leaves.keys()))
        # raise ValueError(f"is: {second_array['spikes'].read()}, should: {np.arange(200)}")
        self.assertTrue(np.all(np.arange(100) == leaves["array"].read()))
        self.assertTrue(np.all(np.arange(200) == second_array["spikes"].read()))

        file.close()

    def test_when_assigning_nested_dict_to_writer_should_write_recursively(self):
        path = "file.h5"
        file = tables.File(path, mode="w")
        file.close()
        file = tables.File(path, mode="r+")

        w = Writer(file, "/")

        w["mydata"] = {"array": np.arange(100), "run_x": {"spikes": np.arange(200)}}

        file.close()

        file = tables.File(path, mode="r")
        nodes, leaves = get_nodes(file, "/mydata")
        _, second_array = get_nodes(file, "/mydata/run_x")

        self.assertEqual(["run_x"], list(nodes.keys()))
        self.assertEqual(["array"], list(leaves.keys()))
        # raise ValueError(f"is: {second_array['spikes'].read()}, should: {np.arange(200)}")
        self.assertTrue(np.all(np.arange(100) == leaves["array"].read()))
        self.assertTrue(np.all(np.arange(200) == second_array["spikes"].read()))

        file.close()

    def test_when_assigning_nested_entries_to_writer_should_write_to_file(self):
        path = "file.h5"
        file = tables.File(path, mode="w")
        file.close()
        file = tables.File(path, mode="r+")

        w = Writer(file, "/")

        w["mydata"] = Node()
        md = w["mydata"]
        md["run_x"] = Node()
        m = md["run_x"]
        m["spikes"] = np.arange(100)

        file.close()

        file = tables.File(path, mode="r")
        _, leaves = get_nodes(file, "/mydata/run_x")
        array = leaves["spikes"].read()
        file.close()

        self.assertTrue(np.all(array == np.arange(100)))

    def test_when_assigning_empty_list_should_persist_as_array(self):
        path = "file2.h5"

        with FileMap(path, mode="write") as w:
            w["a"] = []

        file = tables.File(path, mode="r")
        nodes, leaves = get_nodes(file, "/")
        self.assertTrue(np.all(leaves["a"] == np.array([])))
        file.close()

    def test_when_assigning_list_of_scalars_should_persist_as_array(self):
        path = "file2.h5"

        with FileMap(path, mode="write") as w:
            w["a"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        file = tables.File(path, mode="r")
        nodes, leaves = get_nodes(file, "/")

        self.assertTrue(np.all(leaves["a"] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
        file.close()

    def test_when_assigning_list_of_str_should_persist_as_array(self):
        path = "file2.h5"

        with FileMap(path, mode="write") as w:
            w["a"] = [f"{i}" for i in range(10)]

        file = tables.File(path, mode="r")
        _, leaves = get_nodes(file, "/")
        # raise ValueError(leaves["a"],leaves["a"].read())

        self.assertTrue(
            np.all(
                leaves["a"].read().astype(dtype=str)
                == np.array([f"{i}" for i in range(10)])
            )
        )

        file.close()

    def test_when_assigning_str_should_persist(self):
        path = "file2.h5"
        with FileMap(path, mode="write") as w:
            w["a"] = "abc"
        file = tables.File(path, mode="r")
        _, leaves = get_nodes(file, "/")
        # raise ValueError(leaves["a"],leaves["a"].read())
        self.assertTrue(
            np.all(leaves["a"].read().astype(dtype=str) == np.array(["abc"]))
        )
        file.close()

    def test_when_assigning_raggedly_nested_list_should_raise_value_error(self):
        path = "file2.h5"

        with self.assertRaises(TypeError):
            with FileMap(path, mode="write") as w:
                w["a"] = [0, 2, [0], 2, 3]

    def test_when_assigning_tuple_of_scalars_should_persist_as_array(self):
        path = "file2.h5"

        with FileMap(path, mode="write") as w:
            w["a"] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

        file = tables.File(path, mode="r")
        nodes, leaves = get_nodes(file, "/")
        self.assertTrue(np.all(leaves["a"] == np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))))
        file.close()

    def test_when_assigning_list_of_objects_should_raise_value_error(self):
        path = "file2.h5"

        with self.assertRaises(TypeError):
            with FileMap(path, mode="write") as w:
                w["a"] = [object() for i in range(3)]

    def test_when_assigning_list_of_funcs_should_raise_value_error(self):
        path = "file2.h5"

        def f():
            pass

        x = f

        with self.assertRaises(TypeError):
            with FileMap(path, mode="write") as w:
                w["a"] = [x for i in range(3)]


class TestClassReader(TestCase):
    def _setUp(self):
        path = "file.h5"
        file = tables.File(path, mode="w")
        file.create_group("/", "mydata")
        file.create_group("/mydata", "run_x")
        file.create_array("/mydata/run_x", "spikes", obj=np.arange(100))
        file.close()

    def test_when_reader_key_indexed_should_return_reader_instance(self):
        path = "file.h5"
        file = tables.File(path, mode="r+")
        r = Reader(file, "/")
        self.assertTrue(isinstance(r["mydata"], Reader))
        file.close()

    def test_when_reader_key_indexed_should_return_reader_with_key_appended_to_object_path(
        self,
    ):
        path = "file.h5"
        file = tables.File(path, mode="r+")
        r = Reader(file, "/")
        opath = "/" if r.opath == "/" else r.opath + "/"
        self.assertEqual(opath + "mydata", r["mydata"].opath)
        file.close()

    def test_when_method_keys_called_should_return_nodes_at_respective_level(self):
        path = "file.h5"
        file = tables.File(path, mode="r+")
        r = Reader(file, "/")
        self.assertEqual(["mydata"], list(r.keys()))
        file.close()

    def test_when_leaves_are_retrieved_should_be_of_type_ndarray(self):
        path = "file.h5"
        file = tables.File(path, mode="r+")
        r = Reader(file, "/")
        self.assertTrue(isinstance(r["mydata"]["run_x"]["spikes"], np.ndarray))
        file.close()

    def test_when_leaves_are_returned_should_return_array(self):
        path = "file.h5"
        file = tables.File(path, mode="r+")
        r = Reader(file, "/")
        self.assertTrue(np.all(r["mydata"]["run_x"]["spikes"] == np.arange(100)))
        file.close()

    def test_method_up_when_called_should_move_up_by_one_path_component(self):
        with FileMap("file.h5", mode="read") as r:
            rl = r["mydata"]["run_x"]
            self.assertEqual(rl.up().opath, "/mydata")

    def test_method_up_when_called_on_root_should_raise_opath_error(self):
        with self.assertRaises(opath.OpathError):
            with FileMap("file.h5", mode="read") as r:
                rl = r["mydata"]["run_x"]
                rl.up().up().up()

    def test_method_as_dict_when_called_should_build_dictionary(self):

        with FileMap("file.h5", mode="modify") as w:
            del w["mydata"]["run_x"]["spikes"]
        with FileMap("file.h5", mode="read") as r:
            self.assertEqual(r._as_dict(), {"mydata": {"run_x": {}}})

    def test_when_reading_str_should_return_unicode_python_str(self):
        fname = "file2.h5"
        file = tables.File(fname, "w")
        file.create_array(
            "/", "arr", np.array(["aawjdwjdjwqdk ._?+*jaklHJajkdkwaj", "b", "c"])
        )
        file.close()

        with FileMap(fname, mode="read") as r:
            # raise ValueError(r["arr"], np.array(["aawjdwjdjwqdk ._?+*jaklHJajkdkwaj", "b", "c"]))
            self.assertTrue(
                np.all(
                    r["arr"]
                    == np.array(["aawjdwjdjwqdk ._?+*jaklHJajkdkwaj", "b", "c"])
                )
            )

    def test_method_full_load_when_called_should_read_data_into_nested_dict_structure(
        self,
    ):
        with FileMap("file.h5", mode="read") as r:
            # raise ValueError(r._as_dict(full_load=True)["mydata"]["run_x"]["spikes"] == np.arange(100))
            x = r.full_load()
            self.assertTrue(
                isinstance(x, dict)
                and np.all(x["mydata"]["run_x"]["spikes"] == np.arange(100))
            )

    def test_method_repr_when_str_array_and_long_str_and_short_str_passed_should_return_string_array_and_long_str_shortening_and_full_short_str(
        self,
    ):
        x = "a" * 30 + "b" * 30

        with FileMap("file2.h5", mode="write") as w:
            w["data"] = {"x": "asd", "y": [x for i in range(50)], "z": x}
        with FileMap("file2.h5", mode="read") as r:
            s = r.__repr__()

        should = (
            '{\n  "data": {\n    "x": "asd",\n    "y": "array([\'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...\'\\n '
            + "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n "
            + "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n "
            + "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n "
            + "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n "
            + "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...' ... 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n "
            + "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n "
            + "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n "
            + "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n "
            + "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...'\\n "
            + "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ...']) (50,) dtype:str1408\",\n "
            + '   "z": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbb ..."\n  }\n}'
        )

        # should = '{\n  "data": {\n    "x": "asd",\n    "y": "array([\'aaaaaaaaaaaaaaaaaaaa ...\' \'aaaaaaaaaaaaaaaaaaaa ...\'\\n'\
        # + ' \'aaaaaaaaaaaaaaaaaaaa ...\' \'aaaaaaaaaaaaaaaaaaaa ...\'\\n \'aaaaaaaaaaaaaaaaaaaa ...\' \'aaaaaaaaaaaaaaaaaaaa ...\'\\n'\
        # + ' \'aaaaaaaaaaaaaaaaaaaa ...\' \'aaaaaaaaaaaaaaaaaaaa ...\'\\n \'aaaaaaaaaaaaaaaaaaaa ...\' \'aaaaaaaaaaaaaaaaaaaa ...\' ...'\
        # + ' \'aaaaaaaaaaaaaaaaaaaa ...\' \'aaaaaaaaaaaaaaaaaaaa ...\'\\n \'aaaaaaaaaaaaaaaaaaaa ...\' \'aaaaaaaaaaaaaaaaaaaa ...\'\\n'\
        # + ' \'aaaaaaaaaaaaaaaaaaaa ...\' \'aaaaaaaaaaaaaaaaaaaa ...\'\\n \'aaaaaaaaaaaaaaaaaaaa ...\' \'aaaaaaaaaaaaaaaaaaaa ...\'\\n'\
        # + ' \'aaaaaaaaaaaaaaaaaaaa ...\' \'aaaaaaaaaaaaaaaaaaaa ...\']) (50,) dtype:str768",\n    "z": "aaaaaaaaaaaaaaaaaaaa ..."\n  }\n}'
        # raise ValueError(s, should)
        self.assertEqual(s, should)

    def test_method_repr_when_called_should_return_string_representation(self):
        with FileMap("file2.h5", mode="write") as w:
            w["data"] = {"run_x": {"x": np.arange(100)}}
        with FileMap("file2.h5", mode="read") as r:
            x = r.__repr__()

        self.assertEqual(
            x,
            '{\n  "data": {\n    "run_x": {\n      "x": "array([0 1 2 3 4 5 6 7 8 9 ... 90 91 92 93 94 95 96 97 98 99]) (100,) dtype:int64"\n    }\n  }\n}',
        )

    def test_method_load_when_called_should_return_dictionary_containing_all_descendants_recursively(
        self,
    ):
        path = "file_d.h5"
        file = tables.File(path, mode="w")
        file.create_group("/", "mydata")
        file.create_group("/mydata", "run_x")
        file.create_array("/mydata/run_x", "spikes", obj=np.array([1]))
        file.close()

        file = tables.File(path, mode="r")
        r = Reader(file, "/")
        should = {"mydata": {"run_x": {"spikes": np.array([1])}}}
        is_ = r.load()
        file.close()

        self.assertTrue(is_, should)


class TestClassFileMap(TestCase):
    def test_when_opened_in_write_mode_and_file_does_not_exist_should_write_file(self):
        fname = "file.h5"
        with FileMap(fname, mode="write"):
            pass
        self.assertEqual(os.listdir(), [fname])

    def test_when_opened_in_write_mode_and_file_exists_should_raise_value_error(self):
        fname = "file.h5"
        file = tables.File(fname, "w")
        file.close()
        with self.assertRaises(ValueError):
            FileMap(fname, mode="write")

    def test_when_opened_in_read_mode_should_return_file_contents(self):
        fname = "file.h5"
        file = tables.File(fname, "w")
        file.create_group("/", "group")
        file.close()

        with FileMap(fname) as r:
            self.assertEqual(list(r.keys()), ["group"])

    def test_when_opened_in_write_mode_and_writing_should_raise_error(self):
        fname = "file.h5"
        file = tables.File(fname, "w")
        file.create_group("/", "group")
        file.close()

        fl = None
        with self.assertRaises(ValueError):
            with FileMap(fname, mode="write") as r:
                fl = r.file
                r["a"] = np.array([])
        if fl:
            fl.close()

    def test_when_opened_in_modify_mode_should_not_truncate_file(self):
        fname = "file.h5"
        file = tables.File(fname, "w")
        file.create_group("/", "group")
        file.close()

        with FileMap(fname, mode="modify") as r:
            self.assertEqual(list(r.keys()), ["group"])

    def test_when_opened_in_modify_mode_and_writing_should_append_to_file(self):
        fname = "file.h5"
        file = tables.File(fname, "w")
        file.create_group("/", "group")
        file.close()

        with FileMap(fname, mode="modify") as r:
            r["group2"] = Node()

        fname = "file.h5"
        file = tables.File(fname, "r")
        nodes, leaves = get_nodes(file, "/")
        self.assertEqual(list(nodes.keys()), ["group", "group2"])
        file.close()

    def test_when_entering_file_map_context_in_mode_read_should_return_class_reader(
        self,
    ):
        fname = "file.h5"
        file = tables.File(fname, "w")
        file.close()

        with FileMap(fname) as r:
            self.assertTrue(isinstance(r, Reader))

    def test_when_entering_file_map_context_in_mode_write_should_return_class_writer(
        self,
    ):
        fname = "file.h5"
        with FileMap(fname, mode="write") as r:
            self.assertTrue(isinstance(r, Writer))

    def test_when_entering_file_map_context_in_mode_modify_should_return_class_writer(
        self,
    ):
        fname = "file.h5"
        file = tables.File(fname, "w")
        file.close()

        with FileMap(fname, mode="modify") as r:
            self.assertTrue(isinstance(r, Writer))

    def test_when_exception_raised_should_handle_exit_then_rethrow_the_exception(self):
        fname = "file.h5"
        with self.assertRaises(ValueError):
            with FileMap(fname, mode="write") as w:
                raise ValueError("a")
