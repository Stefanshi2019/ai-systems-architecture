import math

import pytest
import torch

from engine import Value


# ---------- Forward pass / graph construction (should pass before backward is done) ----------

def test_add_forward():
    assert (Value(2.0) + Value(3.0)).data == 5.0


def test_mul_forward():
    assert (Value(2.0) * Value(3.0)).data == 6.0


def test_add_mul_with_python_scalar():
    assert (Value(2.0) + 3).data == 5.0
    assert (Value(2.0) * 3).data == 6.0


def test_graph_children_and_op():
    a, b = Value(2.0), Value(3.0)
    c = a + b
    d = a * b
    assert c._prev == {a, b} and c._op == '+'
    assert d._prev == {a, b} and d._op == '*'


def test_leaf_has_no_children_and_zero_grad():
    a = Value(1.0)
    assert a._prev == set()
    assert a.grad == 0.0


# ---------- Local backward rules ----------

def test_add_local_backward():
    a, b = Value(2.0), Value(3.0)
    c = a + b
    c.grad = 1.0
    c._backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_mul_local_backward():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    c.grad = 1.0
    c._backward()
    assert a.grad == 3.0  # dc/da = b
    assert b.grad == 2.0  # dc/db = a


def test_local_backward_scales_by_upstream_grad():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    c.grad = 5.0
    c._backward()
    assert a.grad == 15.0
    assert b.grad == 10.0


def test_local_backward_accumulates():
    # Gradients must be += not =, otherwise reused nodes get overwritten.
    a, b = Value(2.0), Value(3.0)
    c = a + b
    c.grad = 1.0
    c._backward()
    c._backward()
    assert a.grad == 2.0


# ---------- Full backward() ----------

def test_backward_seeds_output_grad():
    a = Value(2.0)
    out = a + Value(1.0)
    out.backward()
    assert out.grad == 1.0


def test_backward_simple_expression():
    # L = (a * b) + c
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    L = a * b + c
    L.backward()
    assert L.data == 4.0
    assert a.grad == -3.0
    assert b.grad == 2.0
    assert c.grad == 1.0


def test_backward_node_used_twice():
    # y = a * a + a  ->  dy/da = 2a + 1. Tests accumulation across multiple paths.
    a = Value(3.0)
    y = a * a + a
    y.backward()
    assert y.data == 12.0
    assert a.grad == 7.0


def test_backward_diamond_graph():
    # Shared intermediate: each node's _backward must run exactly once, after all its consumers.
    # d = (a + b) * (a + b)  ->  dd/da = dd/db = 2(a+b)
    a, b = Value(1.0), Value(2.0)
    s = a + b
    d = s * s
    d.backward()
    assert d.data == 9.0
    assert s.grad == 6.0
    assert a.grad == 6.0
    assert b.grad == 6.0


def test_backward_deep_chain_ordering():
    # Linear chain forces correct reverse-topological order: x -> *2 -> *2 -> *2 -> +x
    x = Value(1.5)
    y = x
    for _ in range(3):
        y = y * 2
    y = y + x
    y.backward()
    assert y.grad == 1.0
    assert x.grad == 9.0  # 2^3 + 1


def test_backward_with_scalar_operand():
    a = Value(4.0)
    out = a * 3 + 2
    out.backward()
    assert out.data == 14.0
    assert a.grad == 3.0


# ---------- Cross-check against PyTorch autograd ----------

def test_matches_torch_on_composite_expression():
    vals = {"a": 2.0, "b": -3.0, "c": 10.0, "d": 0.5}

    a, b, c, d = (Value(vals[k]) for k in "abcd")
    out = (a * b + c) * d + a * a
    out.backward()

    ta, tb, tc, td = (torch.tensor(vals[k], dtype=torch.float64, requires_grad=True) for k in "abcd")
    tout = (ta * tb + tc) * td + ta * ta
    tout.backward()

    assert out.data == pytest.approx(tout.item())
    for mine, ref in ((a, ta), (b, tb), (c, tc), (d, td)):
        assert mine.grad == pytest.approx(ref.grad.item())


# ======================= Extended ops =======================

# ---------- Reflected operators (Python scalar on the left) ----------

def test_radd():
    a = Value(2.0)
    out = 3 + a
    out.backward()
    assert out.data == 5.0
    assert a.grad == 1.0


def test_rmul():
    a = Value(2.0)
    out = 3 * a
    out.backward()
    assert out.data == 6.0
    assert a.grad == 3.0


# ---------- Negation and subtraction ----------

def test_neg():
    a = Value(2.0)
    out = -a
    out.backward()
    assert out.data == -2.0
    assert a.grad == -1.0


def test_sub():
    a, b = Value(5.0), Value(2.0)
    out = a - b
    out.backward()
    assert out.data == 3.0
    assert a.grad == 1.0
    assert b.grad == -1.0


def test_rsub():
    a = Value(2.0)
    out = 5 - a
    out.backward()
    assert out.data == 3.0
    assert a.grad == -1.0


# ---------- Power (constant int/float exponent) ----------

def test_pow_int():
    a = Value(2.0)
    out = a ** 3
    out.backward()
    assert out.data == 8.0
    assert a.grad == 12.0  # 3 * a^2


def test_pow_fractional():
    a = Value(4.0)
    out = a ** 0.5
    out.backward()
    assert out.data == pytest.approx(2.0)
    assert a.grad == pytest.approx(0.25)  # 0.5 * a^-0.5


def test_pow_negative_exponent():
    a = Value(2.0)
    out = a ** -1
    out.backward()
    assert out.data == pytest.approx(0.5)
    assert a.grad == pytest.approx(-0.25)  # -a^-2


# ---------- Division ----------

def test_truediv():
    a, b = Value(6.0), Value(2.0)
    out = a / b
    out.backward()
    assert out.data == pytest.approx(3.0)
    assert a.grad == pytest.approx(0.5)    # 1/b
    assert b.grad == pytest.approx(-1.5)   # -a/b^2


def test_rtruediv():
    a = Value(2.0)
    out = 6 / a
    out.backward()
    assert out.data == pytest.approx(3.0)
    assert a.grad == pytest.approx(-1.5)   # -6/a^2


# ---------- Nonlinearities ----------

def test_relu_positive():
    a = Value(3.0)
    out = a.relu()
    out.backward()
    assert out.data == 3.0
    assert a.grad == 1.0


def test_relu_negative():
    a = Value(-2.0)
    out = a.relu()
    out.backward()
    assert out.data == 0.0
    assert a.grad == 0.0


def test_tanh():
    a = Value(0.5)
    out = a.tanh()
    out.backward()
    t = math.tanh(0.5)
    assert out.data == pytest.approx(t)
    assert a.grad == pytest.approx(1 - t ** 2)


def test_exp():
    a = Value(1.5)
    out = a.exp()
    out.backward()
    assert out.data == pytest.approx(math.exp(1.5))
    assert a.grad == pytest.approx(math.exp(1.5))


# ---------- Cross-check against PyTorch ----------

def test_extended_ops_match_torch():
    vals = {"a": 1.2, "b": -0.7, "c": 2.5, "d": 1.8}

    a, b, c, d = (Value(vals[k]) for k in "abcd")
    out = ((a * b - c) / d).tanh() + (a ** 2) + (-b).exp() + (c - a).relu() + 2 / d
    out.backward()

    ta, tb, tc, td = (torch.tensor(vals[k], dtype=torch.float64, requires_grad=True) for k in "abcd")
    tout = ((ta * tb - tc) / td).tanh() + (ta ** 2) + (-tb).exp() + (tc - ta).relu() + 2 / td
    tout.backward()

    assert out.data == pytest.approx(tout.item())
    for mine, ref in ((a, ta), (b, tb), (c, tc), (d, td)):
        assert mine.grad == pytest.approx(ref.grad.item())
