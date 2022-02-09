from test.utils import TestCase
import parse_equations


class TestFctEvaluateExpression(TestCase):
    def test_when_passing_augmented_assignment_equation_should_add_to_variable(self):
        n = 10.0

        eqs = f"\nv += {n}*alphax*scale\n"
        parameters = {"alphax": 1.0, "scale": 2.0, "v": 1.0}

        result = parse_equations.evaluate_equations(eqs, parameters)

        self.assertEqual(
            result["v"],
            parameters["v"] + n * parameters["alphax"] * parameters["scale"],
        )

    def test_when_passing_nested_multi_assignment_should_set_variables(self):
        x, a, b = 10, 20, 30
        eqs = f"\n(x,a),b=({x},{a}),{b}\n"

        result = parse_equations.evaluate_equations(eqs, {})

        self.assertEqual(result["x"], x)
        self.assertEqual(result["a"], a)
        self.assertEqual(result["b"], b)

    def test_when_passing_multi_assignment_should_set_variables(self):
        a, b = (
            10,
            20,
        )
        eqs = f"\na,b ={a}, {b}\n"

        result = parse_equations.evaluate_equations(eqs, {})

        self.assertEqual(result["a"], a)
        self.assertEqual(result["b"], b)

    def test_when_multiplying_and_adding_should_evaluate(self):
        a, b, c = (
            10,
            20,
            30,
        )
        eqs = f"\nx = {a} * {b} + {c}\n"

        result = parse_equations.evaluate_equations(eqs, {})

        self.assertEqual(result["x"], a * b + c)

    def test_when_deviding_and_adding_should_evaluate(self):
        a, b, c = (
            10.0,
            20,
            30,
        )
        eqs = f"\nx = {a} / {b} - {c}\n"

        result = parse_equations.evaluate_equations(eqs, {})

        self.assertEqual(result["x"], a / b - c)

    def test_when_power_should_evaluate(self):
        x, p = (
            10,
            2,
        )
        eqs = f"\nx = {x} ** {p}\n"

        result = parse_equations.evaluate_equations(eqs, {})

        self.assertEqual(result["x"], x ** p)

    def test_when_providing_successive_equations_should_be_able_to_resolve_names_from_preceeding_equations(
        self,
    ):
        a, b, c = (
            10.0,
            20,
            30,
        )
        eqs = f"\nx = {a} / {b} - {c}\ny = x ** 2"

        result = parse_equations.evaluate_equations(eqs, {})

        self.assertEqual(result["y"], (a / b - c) ** 2)

    def test_when_assigning_multiple_times_to_same_name_should_overwrite_with_last_equation_assigning_to_respective_name(
        self,
    ):
        a, b, c = (
            10.0,
            20,
            30,
        )
        eqs = f"\nx = {a} / {b} - {c}\nx = 100"

        result = parse_equations.evaluate_equations(eqs, {})

        self.assertEqual(result["x"], 100)


class TestFctExtractVariablesFromEquation(TestCase):
    def test_when_passing_an_add_augmented_assignment_expression_should_extract_variable(
        self,
    ):
        s = "from bla import blub\nimport ast\n(x,a),b=(10,20),30\na, b = (10, 20)\na=10\nx +=1\nx = 10 * 20 + 30\nx = 10.0 / 20 - 30\nx=10**2\nv = 10.0*alphax*scale \n"
        result = parse_equations.extract_variables_from_equations(s)

        self.assertTrue("x" in result.keys())
        self.assertEqual(result["x"], {"binop": "add", "neutral_element": 0})

    def test_when_passing_a_mul_augmented_assignment_expression_should_extract_variable(
        self,
    ):
        s = "from bla import blub\nimport ast\n(x,a),b=(10,20),30\na, b = (10, 20)\na=10\nx =1\nx = 10 * 20 + 30\nx = 10.0 / 20 - 30\nx=10**2\nv *= 10.0*alphax*scale \n"
        result = parse_equations.extract_variables_from_equations(s)

        self.assertTrue("v" in result.keys())
        self.assertEqual(result["v"], {"binop": "mult", "neutral_element": 1})
