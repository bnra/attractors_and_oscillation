from test.utils import TestCase
import persistence


class TestFctVerify(TestCase):
    def test_when_single_component_alpha_path_passed_should_return_empty_str(self):
        self.assertEqual(
            persistence.opath.verify("abc", path_type="single_component"), ""
        )

    def test_when_multi_component_alpha_path_passed_should_return_empty_str(self):
        self.assertEqual(
            persistence.opath.verify("abc/def/ghi", path_type="rel_path"), ""
        )

    def test_when_multi_component_alpha_numeric_underscore_path_passed_should_return_empty_str(
        self,
    ):
        self.assertEqual(
            persistence.opath.verify("abc/def/anc_0998", path_type="rel_path"), ""
        )

    def test_when_abs_multi_component_alpha_path_passed_should_return_empty_str(self):
        self.assertEqual(
            persistence.opath.verify("/abc/def/ghi", path_type="abs_path"), ""
        )

    def test_when_abs_path_passed_with_path_type_rel_path_should_return_non_empty_str(
        self,
    ):
        self.assertTrue(
            persistence.opath.verify("/abc/def/ghi", path_type="rel_path") != ""
        )

    def test_when_root_passed_as_abs_path_should_return_empty_str(self):
        self.assertEqual(persistence.opath.verify("/", path_type="abs_path"), "")

    def test_when_rel_path_passed_with_path_type_abs_path_should_return_non_empty_str(
        self,
    ):
        self.assertTrue(
            persistence.opath.verify("abc/def/ghi", path_type="abs_path") != ""
        )

    def test_when_rel_multi_component_path_passed_with_path_type_single_component_should_return_non_empty_str(
        self,
    ):
        self.assertTrue(
            persistence.opath.verify("abc/def/ghi", path_type="single_component") != ""
        )


class TestFctSplit(TestCase):
    def test_when_relative_path_passed_should_return_components(self):
        self.assertEqual(
            persistence.opath.split("abc/d9_f/ghi"), ["abc", "d9_f", "ghi"]
        )

    def test_when_absolute_path_passed_should_return_components(self):
        self.assertEqual(
            persistence.opath.split("/abc/d9_f/ghi"), ["/", "abc", "d9_f", "ghi"]
        )


class TestFctJoin(TestCase):
    def test_when_single_path_components_passed_should_return_compound_path(self):
        self.assertEqual(persistence.opath.join("abc", "d9_f", "ghi"), "abc/d9_f/ghi")

    def test_when_path_components_passed_and_first_is_abs_should_return_compound_path(
        self,
    ):
        self.assertEqual(persistence.opath.join("/abc", "d9_f", "ghi"), "/abc/d9_f/ghi")
