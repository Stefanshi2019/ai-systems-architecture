"""Thorough tests for practice_engine.Value.

Three independent oracles:
  1. torch.autograd on the same expression (expressions are written once and run on both).
  2. Central finite differences on the engine's own forward pass.
  3. Structural checks on the graph and on the order in which _backward runs.
"""
import math

import pytest
import torch

from practice_engine import Value


# ======================= expressions that run on Value and on torch tensors =======================

def e_add(a, b): return a + b
def e_mul(a, b): return a * b
def e_sub(a, b): return a - b
def e_div(a, b): return a / b
def e_pow_int(a, b): return a ** 3 + b
def e_pow_frac(a, b): return a ** 0.5 * b
def e_pow_neg(a, b): return a ** -2 + b
def e_neg(a, b): return -a * b
def e_radd(a, b): return 3 + a * b
def e_rsub(a, b): return 5 - a * b
def e_rmul(a, b): return 3 * a - b
def e_rdiv(a, b): return 6 / a + b
def e_relu_pos(a, b): return (a * b).relu() + a
def e_relu_neg(a, b): return (a * -b).relu() + a
def e_tanh(a, b): return (a * b).tanh() * b
def e_exp(a, b): return (a - b).exp() * a
def e_reuse(a, b): return a * a + a * b + b * b + a
def e_diamond(a, b):
    s = a + b
    return s * s
def e_shared_subexpr(a, b):
    s = a * b
    return s.tanh() + s ** 2 + s.exp()
def e_div_chain(a, b): return (a / b) / (b / a)
def e_neuron(a, b):
    # classic micrograd-style neuron: tanh(w1*x1 + w2*x2 + bias)
    return (a * 0.5 + b * -1.5 + 0.25).tanh()
def e_kitchen_sink(a, b):
    return ((a * b - a) / (b * b + 1)).tanh() + (a ** 2) + (-b).exp() + (b - a).relu() + 2 / (a * a + 1)


CASES = [
    (e_add, (2.0, -3.0)),
    (e_mul, (2.0, -3.0)),
    (e_sub, (2.0, -3.0)),
    (e_div, (2.0, -3.0)),
    (e_pow_int, (1.5, 0.2)),
    (e_pow_frac, (4.0, 0.7)),
    (e_pow_neg, (1.5, 0.2)),
    (e_neg, (2.0, -3.0)),
    (e_radd, (2.0, -3.0)),
    (e_rsub, (2.0, -3.0)),
    (e_rmul, (2.0, -3.0)),
    (e_rdiv, (2.0, -3.0)),
    (e_relu_pos, (2.0, 3.0)),
    (e_relu_neg, (2.0, 3.0)),   # relu input is negative
    (e_tanh, (0.4, -1.1)),
    (e_exp, (0.7, 0.2)),
    (e_reuse, (1.3, -0.8)),
    (e_diamond, (1.0, 2.0)),
    (e_shared_subexpr, (0.3, 0.9)),
    (e_div_chain, (1.7, 0.6)),
    (e_neuron, (0.8, -0.4)),
    (e_kitchen_sink, (1.2, -0.7)),
]
IDS = [fn.__name__ for fn, _ in CASES]


def run_value(fn, xs):
    vs = [Value(x) for x in xs]
    out = fn(*vs)
    out.backward()
    return out, vs


def run_torch(fn, xs):
    ts = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in xs]
    out = fn(*ts)
    out.backward()
    return out, ts


# ======================= 1. torch oracle =======================

@pytest.mark.parametrize("fn,xs", CASES, ids=IDS)
def test_forward_matches_torch(fn, xs):
    out, _ = run_value(fn, xs)
    ref, _ = run_torch(fn, xs)
    assert out.data == pytest.approx(ref.item())


@pytest.mark.parametrize("fn,xs", CASES, ids=IDS)
def test_gradients_match_torch(fn, xs):
    _, vs = run_value(fn, xs)
    _, ts = run_torch(fn, xs)
    for v, t in zip(vs, ts):
        assert v.grad == pytest.approx(t.grad.item(), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.5, 2.0])
def test_relu_gradient_gate(x):
    a = Value(x)
    out = a.relu() * 7
    out.backward()
    assert out.data == (7 * x if x > 0 else 0)
    assert a.grad == (7.0 if x > 0 else 0.0)


def test_relu_at_zero_blocks_gradient():
    a = Value(0.0)
    out = a.relu()
    out.backward()
    assert out.data == 0.0
    assert a.grad == 0.0


# ======================= 2. finite-difference oracle =======================

def numeric_grad(fn, xs, eps=1e-6):
    grads = []
    for i in range(len(xs)):
        hi, lo = list(xs), list(xs)
        hi[i] += eps
        lo[i] -= eps
        f_hi = fn(*[Value(x) for x in hi]).data
        f_lo = fn(*[Value(x) for x in lo]).data
        grads.append((f_hi - f_lo) / (2 * eps))
    return grads


@pytest.mark.parametrize("fn,xs", CASES, ids=IDS)
def test_gradients_match_finite_differences(fn, xs):
    _, vs = run_value(fn, xs)
    for v, g in zip(vs, numeric_grad(fn, xs)):
        assert v.grad == pytest.approx(g, rel=1e-4, abs=1e-6)


# ======================= 3. graph structure =======================

def test_leaf_attributes():
    a = Value(1.5, label='x')
    assert a.data == 1.5
    assert a.grad == 0.0
    assert a._prev == set()
    assert a._op == ''
    assert a.label == 'x'
    a._backward()  # a leaf's _backward must be callable and harmless
    assert a.grad == 0.0


@pytest.mark.parametrize("make,op", [
    (lambda a, b: a + b, '+'),
    (lambda a, b: a * b, '*'),
])
def test_binary_op_records_graph(make, op):
    a, b = Value(2.0), Value(3.0)
    out = make(a, b)
    assert out._prev == {a, b}
    assert out._op == op


def test_unary_ops_record_single_input():
    a = Value(0.5)
    for out in (a.tanh(), a.exp(), a.relu()):
        assert out._prev == {a}


def test_python_scalar_operands_become_leaf_nodes():
    a = Value(2.0)
    out = a * 3
    assert len(out._prev) == 2
    assert a in out._prev
    (const,) = out._prev - {a}
    assert isinstance(const, Value) and const.data == 3


def test_forward_does_not_mutate_inputs():
    a, b = Value(2.0), Value(3.0)
    _ = a * b + a / b - (-a) + a ** 2
    assert (a.data, b.data) == (2.0, 3.0)
    assert (a.grad, b.grad) == (0.0, 0.0)


def test_builtin_sum_works():
    xs = [Value(float(i)) for i in range(1, 6)]
    total = sum(xs)           # starts from int 0, so needs __radd__
    total.backward()
    assert total.data == 15.0
    assert all(x.grad == 1.0 for x in xs)


def test_independent_graphs_do_not_interfere():
    a, b = Value(2.0), Value(3.0)
    out1 = a * b
    out2 = a + b
    out1.backward()
    assert (a.grad, b.grad) == (3.0, 2.0)
    out2.backward()  # seeds only out2 and walks only out2's graph; grads on a, b accumulate
    assert (a.grad, b.grad) == (4.0, 3.0)


# ======================= 4. backward() ordering contract =======================

def reachable(root):
    seen, stack = {}, [root]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen[n] = None
        stack.extend(n._prev)
    return list(seen)


def instrument(root):
    """Wrap every node's _backward so we can see when (and how often) it runs."""
    nodes = reachable(root)
    calls = []
    for n in nodes:
        orig = n._backward

        def wrapper(n=n, orig=orig):
            calls.append(n)
            orig()
        n._backward = wrapper
    return nodes, calls


def build_wide_graph():
    a, b = Value(1.1), Value(-0.6)
    s = a * b
    t = s + a
    u = s * t
    v = u.tanh() + t * s
    return v


def test_each_interior_node_runs_exactly_once():
    root = build_wide_graph()
    nodes, calls = instrument(root)
    root.backward()
    for n in nodes:
        n_calls = sum(1 for c in calls if c is n)
        if n._prev:
            assert n_calls == 1, f"{n._op} node ran _backward {n_calls} times"
        else:
            assert n_calls <= 1


def test_node_runs_only_after_all_its_consumers():
    root = build_wide_graph()
    nodes, calls = instrument(root)
    consumers = {n: [] for n in nodes}
    for n in nodes:
        for p in n._prev:
            consumers[p].append(n)
    root.backward()
    position = {id(n): i for i, n in enumerate(calls)}
    for n in nodes:
        if id(n) not in position:
            continue  # leaves may be skipped
        for c in consumers[n]:
            assert position[id(c)] < position[id(n)], f"{n._op} ran before its consumer {c._op}"


def test_backward_seeds_with_one_not_overwrites_other_nodes():
    a = Value(2.0)
    mid = a * 3
    out = mid + 1
    out.backward()
    assert out.grad == 1.0
    assert mid.grad == 1.0
    assert a.grad == 3.0


# ======================= 5. stretch goals =======================

def test_stretch_deep_chain_no_recursion_error():
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y * 1.0 + 0.0
    y.backward()
    assert y.data == 1.0
    assert x.grad == 1.0


def test_stretch_wide_fan_in():
    xs = [Value(0.001 * i) for i in range(1, 2001)]
    total = xs[0]
    for x in xs[1:]:
        total = total + x
    total.backward()
    assert all(x.grad == 1.0 for x in xs)
