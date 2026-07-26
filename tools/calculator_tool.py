import ast
import operator

from tools.base_tool import BaseTool

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](_safe_eval(node.operand))
    raise ValueError("表达式包含不支持的语法")


class CalculatorTool(BaseTool):

    @property
    def name(self):
        return "calculator"

    def run(self, expression):
        if not expression or not isinstance(expression, str):
            raise ValueError("表达式不能为空")
        tree = ast.parse(expression.strip(), mode="eval")
        return _safe_eval(tree.body)