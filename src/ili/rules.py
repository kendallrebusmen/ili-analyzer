"""Safe three-valued condition language for the repair-criteria rules (spec section 6.8).

Conditions in ``config/repair_criteria.yaml`` are short text expressions such as
``"0 < depth_pct <= 6 and feature_type in ('dent',)"``. They are parsed with
``ast.parse(text, mode="eval")`` only; the text is never executed. The whole tree is checked
against a whitelist when the condition is compiled, and a small recursive interpreter walks the
validated tree when it is evaluated.

Whitelist (anything else is a ``ValueError`` naming the construct):

* ``BoolOp`` (``and`` / ``or``), ``UnaryOp`` (``not``, unary ``-`` / ``+``)
* ``BinOp`` (``+ - * /``)
* ``Compare`` (``< <= > >= == != in not in``), chains allowed
* ``Name`` from the caller's set of allowed names
* ``Constant`` of type int, float, bool, str or None
* ``Tuple`` / ``List`` of constants, only as the right-hand side of ``in`` / ``not in``

Text longer than ``MAX_CONDITION_CHARS`` or nesting deeper than ``MAX_DEPTH`` is rejected.

Evaluation is Kleene three-valued logic, so a missing input can never make a repair rule pass
silently. ``UNKNOWN`` (None) stands for "not known"; None, NaN (float or numpy) and pandas
NA/NaT inputs are all unknown.

* Arithmetic or a comparison with an unknown operand is unknown; so is division by zero and an
  arithmetic result that is NaN.
* ``and``: any False gives False, else any unknown gives unknown, else True.
* ``or``: any True gives True, else any unknown gives unknown, else False.
* ``not`` unknown is unknown.
* A chain ``a < b <= c`` is ``(a < b) and (b <= c)`` under Kleene ``and``.
* ``x in (c1, c2)`` is ``x == c1 or x == c2`` under Kleene ``or``.

Every operand is evaluated (there is no side effect to skip), so the outcome never depends on
evaluation order. Type errors are configuration errors, not missing data, and raise
``ValueError`` naming the condition: comparing values of different kinds (number, str, bool),
ordering booleans, arithmetic on str or bool, a non-boolean operand of ``and``/``or``/``not``,
or a condition whose result is not boolean. A name the condition references but the caller does
not supply is a programming error (``ValueError``), distinct from a supplied None (unknown).

DRAFT – for review by qualified personnel.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Mapping

import numpy as np
import pandas as pd

UNKNOWN = None  # the third truth value, and the value of any unknown quantity

MAX_CONDITION_CHARS = 500  # spec 6.8: conditions are short; longer text is a configuration error
MAX_DEPTH = 32  # deepest allowed AST nesting, counted from the expression body at depth 1

_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv}
_ORDERING = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
_EQUALITY = {ast.Eq: operator.eq, ast.NotEq: operator.ne}
_MEMBERSHIP = (ast.In, ast.NotIn)
_UNARY = (ast.Not, ast.USub, ast.UAdd)
_BOOL_OPS = (ast.And, ast.Or)
_CONSTANT_TYPES = (bool, int, float, str, type(None))


# --- values -------------------------------------------------------------------------------------


def _normalize(value: Any) -> Any:
    """Map an input to bool, int, float, str or None; None, NaN and pandas NA/NaT become None.

    numpy scalars become the matching Python scalar. Any other type raises ``TypeError``.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return UNKNOWN
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return UNKNOWN
    if isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported value type {type(value).__name__}")


def _kind(value: Any) -> str:
    """The comparison kind of a known value: 'bool', 'number' or 'str'."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return "str"


def _require_number(value: Any, what: str) -> None:
    """Raise ``TypeError`` unless ``value`` is a non-bool number (flags never take arithmetic)."""
    if _kind(value) != "number":
        raise TypeError(f"{what} needs numbers, got {type(value).__name__}")


def _require_truth(value: Any, what: str) -> None:
    """Raise ``TypeError`` unless ``value`` is True, False or unknown."""
    if value is not UNKNOWN and not isinstance(value, bool):
        raise TypeError(f"{what} needs a boolean, got {type(value).__name__}")


def _kleene_and(values: list) -> bool | None:
    """Kleene conjunction: any False → False, else any unknown → unknown, else True."""
    if any(v is False for v in values):
        return False
    if any(v is UNKNOWN for v in values):
        return UNKNOWN
    return True


def _kleene_or(values: list) -> bool | None:
    """Kleene disjunction: any True → True, else any unknown → unknown, else False."""
    if any(v is True for v in values):
        return True
    if any(v is UNKNOWN for v in values):
        return UNKNOWN
    return False


def _kleene_not(value: bool | None) -> bool | None:
    """Kleene negation: not unknown is unknown."""
    return UNKNOWN if value is UNKNOWN else not value


# --- validation ---------------------------------------------------------------------------------


def _check_constant(node: ast.Constant) -> None:
    """Allow only int, float, bool, str and None literals."""
    if not isinstance(node.value, _CONSTANT_TYPES):
        raise ValueError(f"constant of type {type(node.value).__name__} is not allowed")


def _check_literal_sequence(node: ast.AST) -> None:
    """The right side of ``in``: a Tuple or List whose elements are all constants."""
    if not isinstance(node, (ast.Tuple, ast.List)):
        raise ValueError(f"'in' needs a literal tuple or list on the right, got {type(node).__name__}")
    for element in node.elts:
        if not isinstance(element, ast.Constant):
            raise ValueError(f"'in' list element {type(element).__name__} is not a constant")
        _check_constant(element)


def _check_compare(node: ast.Compare, names: set[str], refs: set[str], depth: int) -> None:
    """Validate a (possibly chained) comparison; a literal list may only follow ``in``."""
    _check(node.left, names, refs, depth + 1)
    last = len(node.ops) - 1
    for i, (op, right) in enumerate(zip(node.ops, node.comparators)):
        if isinstance(op, _MEMBERSHIP):
            if i != last:
                raise ValueError("a literal 'in' list cannot start another comparison")
            _check_literal_sequence(right)
        elif type(op) in _ORDERING or type(op) in _EQUALITY:
            _check(right, names, refs, depth + 1)
        else:
            raise ValueError(f"comparison operator {type(op).__name__} is not allowed")


def _check(node: ast.AST, names: set[str], refs: set[str], depth: int) -> None:
    """Recursively validate ``node`` against the whitelist, collecting referenced names."""
    if depth > MAX_DEPTH:
        raise ValueError(f"nesting depth exceeds MAX_DEPTH={MAX_DEPTH}")
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _BOOL_OPS):
            raise ValueError(f"boolean operator {type(node.op).__name__} is not allowed")
        for value in node.values:
            _check(value, names, refs, depth + 1)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _UNARY):
            raise ValueError(f"unary operator {type(node.op).__name__} is not allowed")
        _check(node.operand, names, refs, depth + 1)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _ARITHMETIC:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        _check(node.left, names, refs, depth + 1)
        _check(node.right, names, refs, depth + 1)
    elif isinstance(node, ast.Compare):
        _check_compare(node, names, refs, depth)
    elif isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"unknown name {node.id!r}")
        refs.add(node.id)
    elif isinstance(node, ast.Constant):
        _check_constant(node)
    else:
        raise ValueError(f"{type(node).__name__} is not allowed")


# --- evaluation ---------------------------------------------------------------------------------


def _arithmetic(op: ast.operator, left: Any, right: Any) -> Any:
    """Numeric ``+ - * /``; unknown operand, division by zero or a NaN result gives unknown."""
    if left is UNKNOWN or right is UNKNOWN:
        return UNKNOWN
    name = type(op).__name__
    _require_number(left, name)
    _require_number(right, name)
    if isinstance(op, ast.Div) and right == 0:
        return UNKNOWN
    result = _ARITHMETIC[type(op)](left, right)
    return UNKNOWN if isinstance(result, float) and math.isnan(result) else result


def _compare_pair(op: ast.cmpop, left: Any, right: Any) -> bool | None:
    """One link of a chain; unknown operand gives unknown, different kinds raise ``TypeError``."""
    if left is UNKNOWN or right is UNKNOWN:
        return UNKNOWN
    if _kind(left) != _kind(right):
        raise TypeError(f"cannot compare {type(left).__name__} with {type(right).__name__}")
    if type(op) in _ORDERING:
        if _kind(left) == "bool":
            raise TypeError("booleans cannot be ordered")
        return _ORDERING[type(op)](left, right)
    return _EQUALITY[type(op)](left, right)


def _membership(op: ast.cmpop, left: Any, elements: list) -> bool | None:
    """``in`` as a Kleene ``or`` of equalities; ``not in`` is its Kleene negation."""
    found = _kleene_or([_compare_pair(ast.Eq(), left, e) for e in elements])
    return _kleene_not(found) if isinstance(op, ast.NotIn) else found


def _value(node: ast.AST, variables: Mapping[str, Any]) -> Any:
    """Interpret a validated node; returns a known value or ``UNKNOWN``."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _normalize(variables[node.id])
    if isinstance(node, ast.BoolOp):
        values = [_value(v, variables) for v in node.values]
        for v in values:
            _require_truth(v, type(node.op).__name__)
        return _kleene_and(values) if isinstance(node.op, ast.And) else _kleene_or(values)
    if isinstance(node, ast.UnaryOp):
        operand = _value(node.operand, variables)
        if isinstance(node.op, ast.Not):
            _require_truth(operand, "Not")
            return _kleene_not(operand)
        if operand is UNKNOWN:
            return UNKNOWN
        _require_number(operand, type(node.op).__name__)
        return -operand if isinstance(node.op, ast.USub) else +operand
    if isinstance(node, ast.BinOp):
        return _arithmetic(node.op, _value(node.left, variables), _value(node.right, variables))
    if isinstance(node, ast.Compare):
        links = []
        left = _value(node.left, variables)
        for op, right_node in zip(node.ops, node.comparators):
            if isinstance(op, _MEMBERSHIP):
                links.append(_membership(op, left, [e.value for e in right_node.elts]))
                break  # validation guarantees a membership link is the last one
            right = _value(right_node, variables)
            links.append(_compare_pair(op, left, right))
            left = right
        return _kleene_and(links)
    raise AssertionError(f"unvalidated node {type(node).__name__}")  # pragma: no cover


# --- public interface ---------------------------------------------------------------------------


class Condition:
    """A compiled, validated condition.

    ``text`` is the source, ``names`` the frozenset of variable names it references.
    """

    def __init__(self, text: str, tree: ast.Expression, names: frozenset[str]):
        self.text = text
        self.names = names
        self._tree = tree

    def __repr__(self) -> str:
        return f"Condition({self.text!r})"

    def evaluate(self, variables: Mapping[str, Any]) -> bool | None:
        """True, False or ``UNKNOWN`` (None) under Kleene logic for the given variables.

        Every referenced name must be a key of ``variables`` (its value may be None/NaN, which
        is unknown). A missing name or a type error raises ``ValueError`` naming the condition.
        """
        missing = sorted(self.names - set(variables))
        if missing:
            raise ValueError(f"condition {self.text!r}: missing variable(s) {missing}")
        try:
            result = _value(self._tree.body, variables)
            _require_truth(result, "condition result")
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"condition {self.text!r}: {exc}") from None
        return result


def compile_condition(text: str, names: set[str]) -> Condition:
    """Parse and validate ``text`` against the whitelist; ``names`` are the allowed variables.

    Raises ``ValueError`` naming the offending construct for anything outside the whitelist,
    a syntax error, text over ``MAX_CONDITION_CHARS`` or nesting over ``MAX_DEPTH``.
    """
    if not isinstance(text, str):
        raise ValueError(f"condition must be a string, got {type(text).__name__}")
    if len(text) > MAX_CONDITION_CHARS:
        raise ValueError(
            f"condition is {len(text)} characters; MAX_CONDITION_CHARS is {MAX_CONDITION_CHARS}")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise ValueError(f"condition {text!r} does not parse: {exc}") from None
    refs: set[str] = set()
    try:
        _check(tree.body, set(names), refs, depth=1)
    except ValueError as exc:
        raise ValueError(f"condition {text!r}: {exc}") from None
    return Condition(text, tree, frozenset(refs))
