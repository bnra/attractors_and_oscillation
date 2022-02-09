import ast
from typing import Dict, Any, Tuple
import copy


def evaluate_node(node, parameters):
    """
    evaluate abstract syntax tree node recurviely using params to resolve
    external variables
    """

    def traverse_operator(node):
        value = None
        if isinstance(node, ast.Add):
            value = lambda a, b: a + b
        elif isinstance(node, ast.Mult):
            value = lambda a, b: a * b
        elif isinstance(node, ast.Sub):
            value = lambda a, b: a - b
        elif isinstance(node, ast.Div):
            value = lambda a, b: a / b
        elif isinstance(node, ast.Pow):
            value = lambda a, b: a ** b
        else:
            raise NotImplementedError(f"Cannot deal with {type(node)}.")
        return value

    def resolve(variable):
        if variable in parameters:
            return parameters[variable]
        raise Exception(f"{variable} not in parameters. Cannot be resolved.")

    def eval_exp(binop, left, right):
        if isinstance(left, str):
            left = resolve(left)
        if isinstance(right, str):
            right = resolve(right)
        return binop(left, right)

    def assign(targets, value):
        if isinstance(targets, list):
            for t, v in zip(targets, value):
                assign(t, v)
        elif isinstance(targets, str):
            parameters[targets] = value
        else:
            raise NotImplementedError(f"{type(targets)} not implemented.")

    def evaluate(node):
        value = None
        if isinstance(node, ast.Assign):
            targets, values = evaluate(node.targets), evaluate(node.value)
            assign(targets, values)
        elif isinstance(node, ast.AugAssign):
            value = eval_exp(
                traverse_operator(node.op), evaluate(node.target), evaluate(node.value)
            )
            assign(evaluate(node.target), value)
        elif isinstance(node, list):
            if len(node) == 1:
                value = evaluate(node[0])
            else:
                raise NotImplementedError("Cannot deal with lists of length != 1.")
        elif isinstance(node, ast.Tuple):
            value = []
            for n in node.elts:
                value.append(evaluate(n))
        elif isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Name):
            value = node.id
        elif isinstance(node, ast.BinOp):
            value = eval_exp(
                traverse_operator(node.op), evaluate(node.left), evaluate(node.right)
            )
        else:
            raise NotImplementedError(f"Cannot deal with {type(node)}.")
        return value

    evaluate(node)
    return parameters


def extract_node(node, variables: Dict[str, dict]) -> Tuple[str, int]:
    def extract_op(node):
        if isinstance(node, ast.Add):
            return ("add", 0)
        elif isinstance(node, ast.Mult):
            return ("mult", 1)
        elif isinstance(node, ast.Div):
            return ("div", 1)
        elif isinstance(node, ast.Sub):
            return ("sub", 0)
        else:
            raise NotImplementedError(f"Cannot deal with {type(node)}.")

    def extract(node):
        if isinstance(node, ast.AugAssign):
            binop, neutral = extract_op(node.op)
            variables[extract(node.target)] = {
                "binop": binop,
                "neutral_element": neutral,
            }
        elif isinstance(node, ast.Tuple):
            for n in node.elts:
                extract(n)
        elif isinstance(node, ast.Name):
            return node.id

    extract(node)
    return variables


class EquationEvaluator(ast.NodeVisitor):
    def __init__(self, params):
        self.parameters = copy.deepcopy(params)

    def visit_Assign(self, node):
        evaluate_node(node, self.parameters)

    def visit_AugAssign(self, node):
        evaluate_node(node, self.parameters)

    @property
    def report(self):
        return self.parameters


def evaluate_equations(equations: str, parameters: Dict[str, Any]):
    x = ast.parse(equations)
    analyzer = EquationEvaluator(params=parameters)
    analyzer.visit(x)
    return analyzer.report


class VariableExtractor(ast.NodeVisitor):
    def __init__(self):
        self.variables = {}

    def visit_AugAssign(self, node):
        extract_node(node, self.variables)

    @property
    def report(self):
        return self.variables


def extract_variables_from_equations(equations: str):
    x = ast.parse(equations)
    analyzer = VariableExtractor()
    analyzer.visit(x)
    return analyzer.report
