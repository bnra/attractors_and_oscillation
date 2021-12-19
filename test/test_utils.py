from brian2 import ms, second
import numpy as np
import os

from test.utils import TestCase
from utils import clean_brian2_quantity, get_brian2_base_unit, get_brian2_unit, unique_idx, validate_file_path, generate_sequential_file_name


class TestFctValidateFilePath(TestCase):
    def test_when_relative_path_should_return_empty_error_str(self):
        rel_fname = "abcdefghijklmnop.h5"
        self.assertEqual(validate_file_path(rel_fname, ext=".h5"), "")

    def test_when_absolute_path_should_return_empty_error_str(self):
        rel_fname = "abcdefghijklmnop.h5"
        abs_fname = os.path.join(self.tmp_dir, rel_fname)
        
        self.assertEqual(validate_file_path(abs_fname, ext=".h5"), "")

    def test_when_path_does_not_exist_should_return_non_empty_error_str(self):
        rel_fname = "abcdefghijklmnop.h5"
        fname =  os.path.join(os.path.join(self.tmp_dir, "testin"), rel_fname)
        self.assertNotEqual(validate_file_path(fname, ext=".h5"), "")

    def test_when_provided_ext_does_not_match_should_return_non_empty_error_str(self):
        fname = "a.h5"
        ext = ".h5p"
        self.assertNotEqual(validate_file_path(fname, ext=ext), "")

    def test_when_filename_consists_of_all_groups_of_permissible_literals_should_return_empty_error_str(
        self,
    ):
        fname = "aA0_cZ1_.h5"
        self.assertEqual(validate_file_path(fname, ext=".h5"), "")  # regex test

    def test_when_filename_consists_of_all_groups_of_permissible_literals_should_return_empty_error_str(
        self,
    ):
        # include unwanted chars
        literals = ["-", " ", "ä", "(", "?", "#", "~"]
        names = [
            validate_file_path("abcdejjk" + uw + ".h5", ext=".h5") for uw in literals
        ]
        self.assertNotEqual(names, ["" for i in range(len(literals))])

    def test_when_filename_of_length_one_should_return_empty_error_str(self):
        # file name length test
        fname = "a"
        self.assertEqual(validate_file_path(fname), "")

    def test_when_filename_of_length_256_should_return_non_empty_error_str(
        self,
    ):
        # total of 256 > 255
        fname = "a" * 253 + ".h5"
        self.assertNotEqual(validate_file_path(fname, ext=".h5"), "")

    def test_when_filename_lacks_extension_and_no_extension_specified_should_return_empty_error_str(
        self,
    ):
        # without filename extension
        fname = "abcdefghijklmnop"
        self.assertEqual(validate_file_path(fname), "")

    def test_when_filename_has_extension_and_no_extension_specified_should_return_non_empty_error_str(
        self,
    ):
        # without filename extension
        fname = "abcdefghijklmnop.h5"
        self.assertNotEqual(validate_file_path(fname), "")


class TestFctGenerateSequentialFileName(TestCase):
    def test_when_creating_files_sequentially_should_create_files_with_incrementing_suffixes_starting_at_zero(
        self,
    ):
        base_path = "experiments"
        base_name = "exp"
        ext = ".h5"
        os.makedirs(base_path)
        for i in range(10):
            open(
                generate_sequential_file_name(base_path, base_name, ext), mode="w"
            ).close()

        self.assertEqual(
            sorted(os.listdir(base_path)), [f"{base_name}_{i}{ext}" for i in range(10)]
        )

class TestFctCleanBrian2Quantity(TestCase):
    def test_when_called_should_remove_unit_and_provide_str_repr(self):
        x = np.arange(10) * ms
        x_clean, unit = clean_brian2_quantity(x)
        self.assertTrue(np.all(x_clean == np.arange(10, dtype=float)) and unit == "ms")

class TestFctGetBrian2Unit(TestCase):
    def test_when_called_should_return_unit(self):
        x = np.arange(10) * ms
        unit = get_brian2_unit(x)
        self.assertEqual(unit, ms)

class TestFctGetBrian2BaseUnit(TestCase):
    def test_when_called_should_return_base_unit(self):
        x = np.arange(10) * ms
        unit = get_brian2_base_unit(x)
        self.assertEqual(unit, second)


class TestFctUniqueIdx(TestCase):
    def test_when_called_with_tiled_input_should_return_unique_indices(self):
        repeat = 100
        x = np.tile(np.arange(100), repeat).reshape(repeat, 100)
        y = x.ravel()
        self.assertTrue(np.all(np.asarray([np.sum(y[idx])/(i) if i > 0 else float(len(y[idx])) for i,idx in enumerate(unique_idx(y)[1])]) == float(repeat)))

    def test_when_called_with_arbitrary_input_should_return_unique_indices(self):
        x = np.array([0,0,3,4,7,0,1,0,7,4,7])
        should_idx = [np.asarray(e) for e in [[0,1,5,7], [6], [2], [3,9], [4,8,10]]]

        self.assertTrue(all([np.all(is_== should_) for is_, should_ in zip(unique_idx(x)[1], should_idx)]))



    def test_when_called_with_non_one_dim_input_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            unique_idx(np.arange(100).reshape(10,10))
    
