import inspect
import math
import re

import numpy as np
import pandas as pd
import pytest

import ili.rules
from ili.rules import MAX_CONDITION_CHARS, MAX_DEPTH, UNKNOWN, compile_condition

NAMES = {"a", "b", "c", "t", "x", "y", "p", "q", "r"}


def ev(text, **variables):
    return compile_condition(text, NAMES).evaluate(variables)


# --- arithmetic and comparison ------------------------------------------------------------------


def test_unknown_is_none():
    assert UNKNOWN is None


def test_simple_comparison():
    assert ev("a > 80", a=81) is True
    assert ev("a > 80", a=80) is False


def test_chained_comparison():
    assert ev("0 < a <= 6", a=6) is True
    assert ev("0 < a <= 6", a=0) is False
    assert ev("0 < a <= 6", a=6.5) is False


def test_arithmetic():
    assert ev("a / b * 100 > 3", a=0.04, b=1.0) is True
    assert ev("a / b * 100 > 3", a=0.02, b=1.0) is False
    assert ev("a - b + 1 == 2", a=3, b=2) is True
    assert ev("-a < 0", a=1) is True
    assert ev("+a > 0", a=1) is True


def test_membership():
    assert ev("t in ('gouge','dent')", t="dent") is True
    assert ev("t in ('gouge','dent')", t="metal_loss") is False
    assert ev("t not in ('gouge','dent')", t="metal_loss") is True
    assert ev("t in ['gouge', 'dent']", t="gouge") is True


def test_numpy_values():
    assert ev("a > 80", a=np.float64(81.0)) is True
    assert ev("a > 80", a=np.int64(81)) is True
    assert ev("p", p=np.bool_(True)) is True


def test_names_reports_referenced_variables():
    cond = compile_condition("a > 1 and t in ('x',)", NAMES)
    assert cond.names == frozenset({"a", "t"})
    assert cond.text == "a > 1 and t in ('x',)"


# --- Kleene logic -------------------------------------------------------------------------------

T, F, U = True, False, None

AND_TABLE = [(T, T, T), (T, F, F), (T, U, U), (F, T, F), (F, F, F), (F, U, F),
             (U, T, U), (U, F, F), (U, U, U)]
OR_TABLE = [(T, T, T), (T, F, T), (T, U, T), (F, T, T), (F, F, F), (F, U, U),
            (U, T, T), (U, F, U), (U, U, U)]


@pytest.mark.parametrize("p,q,expected", AND_TABLE)
def test_kleene_and(p, q, expected):
    assert ev("p and q", p=p, q=q) is expected


@pytest.mark.parametrize("p,q,expected", OR_TABLE)
def test_kleene_or(p, q, expected):
    assert ev("p or q", p=p, q=q) is expected


@pytest.mark.parametrize("p,expected", [(T, F), (F, T), (U, U)])
def test_kleene_not(p, expected):
    assert ev("not p", p=p) is expected


def test_deciding_short_circuits():
    assert ev("False and p", p=None) is False
    assert ev("True or p", p=None) is True
    assert ev("p and a > 1", p=False, a=None) is False
    assert ev("p or a > 1", p=True, a=None) is True


def test_three_way_boolops():
    assert ev("p and q and r", p=True, q=None, r=False) is False
    assert ev("p or q or r", p=False, q=None, r=False) is None


def test_unknown_operand_propagates():
    assert ev("a > 80", a=None) is None
    assert ev("a / b * 100 > 3", a=None, b=1.0) is None
    assert ev("t in ('gouge','dent')", t=None) is None
    assert ev("-a < 0", a=None) is None


def test_chained_comparison_is_kleene_and():
    # A False link decides even when another link is unknown.
    assert ev("a < b <= c", a=5, b=1, c=None) is False
    assert ev("a < b <= c", a=None, b=1, c=0) is False
    assert ev("a < b <= c", a=0, b=1, c=None) is None


# --- NaN and division by zero ------------------------------------------------------------------


@pytest.mark.parametrize("nan", [float("nan"), np.float64("nan"), np.float32("nan"), pd.NA])
def test_nan_is_unknown(nan):
    assert ev("a > 80", a=nan) is None
    assert ev("not (a > 80)", a=nan) is None
    assert ev("a > 80 or p", a=nan, p=True) is True


def test_division_by_zero_is_unknown():
    assert ev("a / b > 1", a=1.0, b=0.0) is None
    assert ev("a / b > 1", a=1, b=0) is None
    assert ev("a / b > 1 and p", a=1.0, b=0.0, p=False) is False


def test_nan_arithmetic_result_is_unknown():
    assert ev("a - b > 0", a=math.inf, b=math.inf) is None


# --- configuration errors at evaluate time -----------------------------------------------------


def test_type_mismatch_is_a_configuration_error():
    with pytest.raises(ValueError, match="t < 3"):
        ev("t < 3", t="dent")
    with pytest.raises(ValueError):
        ev("t == 3", t="dent")
    with pytest.raises(ValueError):
        ev("t in (1, 2)", t="dent")
    with pytest.raises(ValueError):
        ev("t + 1 > 0", t="dent")


def test_bool_arithmetic_is_rejected():
    with pytest.raises(ValueError):
        ev("True + 1 > 0")
    with pytest.raises(ValueError):
        ev("p + 1 > 0", p=True)
    with pytest.raises(ValueError):
        ev("-p", p=True)


def test_non_boolean_logic_operand_is_rejected():
    with pytest.raises(ValueError):
        ev("a and p", a=3, p=True)
    with pytest.raises(ValueError):
        ev("a", a=3)


def test_missing_variable_is_an_error_but_none_is_unknown():
    cond = compile_condition("a > 80", NAMES)
    with pytest.raises(ValueError, match="'a'"):
        cond.evaluate({})
    assert cond.evaluate({"a": None}) is None


def test_unsupported_variable_type_is_rejected():
    with pytest.raises(ValueError):
        ev("a > 1", a=[1, 2])


# --- rejections at compile time ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os')",
        "a.b",
        "f(a)",
        "a[0]",
        "[x for x in a]",
        "lambda: 1",
        "a ** 2",
        "x if a else y",
        "zz > 1",  # unknown name
        "a // 2 > 1",
        "a % 2 == 0",
        "a is None",
        "(a, b)",
        "a in b",  # membership needs a literal tuple/list
        "a in (b, 1)",  # tuple elements must be constants
        "a < (1, 2)",  # tuple outside membership
        "a in (1, 2) < 3",  # tuple used as a chain's left operand
        "b'x' == a",
        "1j == a",
        "{a: 1}",
        "f'{a}'",
        "(y := 1)",
        "a > 1 > ",  # syntax error
        "",
    ],
)
def test_rejections(text):
    with pytest.raises(ValueError):
        compile_condition(text, NAMES)


def test_rejects_long_text():
    text = "a > " + "1" * (MAX_CONDITION_CHARS - 3)
    assert len(text) == MAX_CONDITION_CHARS + 1
    with pytest.raises(ValueError, match="500"):
        compile_condition(text, NAMES)
    compile_condition(text[:-1], NAMES)  # exactly 500 is allowed


def test_rejects_deep_nesting():
    deep = "a" + " + 1" * 40 + " > 0"
    assert len(deep) < MAX_CONDITION_CHARS
    with pytest.raises(ValueError, match="depth"):
        compile_condition(deep, NAMES)
    assert MAX_DEPTH == 32
    shallow = "a" + " + 1" * 10 + " > 0"
    assert compile_condition(shallow, NAMES).evaluate({"a": 0}) is True


def test_rejects_non_string():
    with pytest.raises(ValueError):
        compile_condition(42, NAMES)


def test_rejection_names_the_construct():
    with pytest.raises(ValueError, match="Call"):
        compile_condition("f(a)", NAMES)
    with pytest.raises(ValueError, match="Pow"):
        compile_condition("a ** 2", NAMES)
    with pytest.raises(ValueError, match="zz"):
        compile_condition("zz > 1", NAMES)


def test_no_eval_used():
    source = inspect.getsource(ili.rules)
    for banned in ("eval(", "exec(", "compile("):
        assert banned not in source, banned
    calls = re.findall(r"ast\.parse\(", source)
    assert calls, "rules must parse with ast.parse"
    assert 'mode="eval"' in source
