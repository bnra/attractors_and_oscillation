import os

from test.utils import TestCase
from utils import validate_file_path, generate_sequential_file_name


class TestFctValidateFilePath(TestCase):
    def test_when_relative_path_should_return_empty_error_str(self):
        rel_fname = "abcdefghijklmnop.h5"
        self.assertEqual(validate_file_path(rel_fname, ext=".h5"), "")

    def test_when_absolute_path_should_return_empty_error_str(self):
        rel_fname = "abcdefghijklmnop.h5"
        abs_fname = (
            f"/home/ben/Desktop/idp/implementation/repo/{self.tmp_base_name}/"
            + rel_fname
        )
        self.assertEqual(validate_file_path(abs_fname, ext=".h5"), "")

    def test_when_path_does_not_exist_should_return_non_empty_error_str(self):
        rel_fname = "abcdefghijklmnop.h5"
        fname = "home/ben/Desktop/idp/implementation/repo/.testin/" + rel_fname
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

