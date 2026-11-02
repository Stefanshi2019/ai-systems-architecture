"""Tests for tensor_layout. The oracle is PyTorch itself: build a real view, then check that your
pure-Python layout arithmetic predicts exactly the (shape, stride, storage_offset) torch reports.
"""
import itertools
import math

import pytest
import torch

from tensor_layout import (
    contiguous_strides,
    expand_layout,
    flat_index,
    is_contiguous_layout,
    shares_storage,
    slice_layout,
    transpose_layout,
    view_possible,
)


def base(shape):
    """Tensor whose element value equals its position in storage."""
    return torch.arange(math.prod(shape)).reshape(shape)


TENSORS = {
    "plain_2x3x4": lambda: base((2, 3, 4)),
    "transpose": lambda: base((2, 3, 4)).transpose(0, 1),
    "permute": lambda: base((2, 3, 4)).permute(2, 0, 1),
    "row_slice": lambda: base((4, 6))[1:3],
    "col_slice": lambda: base((4, 6))[:, 1:5],
    "step_slice": lambda: base((4, 6))[:, ::2],
    "expand": lambda: base((3, 1)).expand(3, 4),
    "unsqueeze": lambda: base((2, 3)).unsqueeze(1),
    "size1_col": lambda: base((4, 6))[:, 2:3],
    "chained": lambda: base((6, 8))[1:5, ::2].t(),
}
NAMES = list(TENSORS)


def layout(t):
    return tuple(t.shape), tuple(t.stride()), t.storage_offset()


# ======================= strides and addressing =======================

@pytest.mark.parametrize("shape", [(5,), (2, 3), (2, 3, 4), (1, 3), (2, 1, 4), (3, 1, 1), (2, 3, 4, 5)])
def test_contiguous_strides(shape):
    assert tuple(contiguous_strides(shape)) == tuple(torch.empty(shape).stride())


@pytest.mark.parametrize("name", NAMES)
def test_flat_index_finds_the_element(name):
    t = TENSORS[name]()
    offset = t.storage_offset()
    strides = t.stride()
    for idx in itertools.product(*(range(n) for n in t.shape)):
        # base() stores arange, so an element's value is its storage position
        assert flat_index(idx, strides, offset) == t[idx].item()


# ======================= view arithmetic =======================

@pytest.mark.parametrize("shape,d0,d1", [
    ((2, 3), 0, 1),
    ((2, 3, 4), 0, 1),
    ((2, 3, 4), 0, 2),
    ((2, 3, 4), 1, 2),
    ((2, 3, 4), 2, 0),
    ((4, 1, 5), 0, 1),
])
def test_transpose_layout(shape, d0, d1):
    t = base(shape)
    got_shape, got_strides = transpose_layout(tuple(t.shape), tuple(t.stride()), d0, d1)
    ref = t.transpose(d0, d1)
    assert (tuple(got_shape), tuple(got_strides)) == (tuple(ref.shape), tuple(ref.stride()))


def apply_slice(t, dim, start, stop, step):
    return t[(slice(None),) * dim + (slice(start, stop, step),)]


@pytest.mark.parametrize("shape,dim,start,stop,step", [
    ((4, 6), 0, 1, 3, 1),
    ((4, 6), 1, 1, 5, 1),
    ((4, 6), 1, 0, 6, 2),
    ((4, 6), 1, 1, 6, 2),
    ((4, 6), 1, 2, 100, 1),   # stop beyond the end is clamped
    ((4, 6), 0, 3, 100, 3),
    ((2, 3, 4), 2, 1, 3, 1),
    ((2, 3, 4), 1, 0, 3, 2),
])
def test_slice_layout(shape, dim, start, stop, step):
    t = base(shape)
    got = slice_layout(tuple(t.shape), tuple(t.stride()), t.storage_offset(), dim, start, stop, step)
    assert tuple(map(lambda x: tuple(x) if isinstance(x, (tuple, list)) else x, got)) == layout(
        apply_slice(t, dim, start, stop, step)
    )


def test_slice_layout_chained_keeps_accumulating_offset():
    t = base((6, 8))
    s1 = slice_layout(tuple(t.shape), tuple(t.stride()), t.storage_offset(), 0, 1, 5, 1)
    s2 = slice_layout(*s1, 1, 1, 8, 2)
    ref = t[1:5][:, 1:8:2]
    assert tuple(map(lambda x: tuple(x) if isinstance(x, (tuple, list)) else x, s2)) == layout(ref)


@pytest.mark.parametrize("shape,new_shape", [
    ((3, 1), (3, 4)),
    ((1, 4), (5, 4)),
    ((4,), (2, 3, 4)),
    ((2, 1, 3), (2, 5, 3)),
    ((2, 3), (2, 3)),
    ((1,), (7,)),
])
def test_expand_layout(shape, new_shape):
    t = base(shape)
    got_shape, got_strides = expand_layout(tuple(t.shape), tuple(t.stride()), new_shape)
    ref = t.expand(*new_shape)
    assert (tuple(got_shape), tuple(got_strides)) == (tuple(ref.shape), tuple(ref.stride()))


# ======================= contiguity and sharing =======================

@pytest.mark.parametrize("name", NAMES)
def test_is_contiguous_layout(name):
    t = TENSORS[name]()
    assert is_contiguous_layout(tuple(t.shape), tuple(t.stride())) == t.is_contiguous()


def test_contiguity_cases_cover_both_answers():
    answers = {TENSORS[n]().is_contiguous() for n in NAMES}
    assert answers == {True, False}


def test_shares_storage_views_and_copies():
    a = base((4, 6))
    assert shares_storage(a, a)
    assert shares_storage(a, a.view(6, 4))
    assert shares_storage(a, a.t())
    assert shares_storage(a, a[1:3])
    assert shares_storage(a, a[:, ::2])
    assert shares_storage(a, a.reshape(24))               # contiguous: reshape is a view
    assert shares_storage(a, a.contiguous())              # already contiguous: no copy
    assert not shares_storage(a, a.clone())
    assert not shares_storage(a, a.t().contiguous())      # had to materialise a copy
    assert not shares_storage(a, a.t().reshape(24))       # reshape copied
    assert not shares_storage(a, torch.arange(24).reshape(4, 6))


def test_shares_storage_means_writes_are_visible():
    a = base((4, 6))
    v = a.t()
    assert shares_storage(a, v)
    v[0, 1] = 999
    assert a[1, 0].item() == 999


# ======================= stretch: when is .view legal? =======================

VIEW_CASES = [
    ("plain_flatten", lambda: base((2, 3, 4)), (24,)),
    ("plain_merge", lambda: base((2, 3, 4)), (6, 4)),
    ("plain_split", lambda: base((2, 3, 4)), (2, 3, 2, 2)),
    ("transpose_flatten", lambda: base((2, 3, 4)).transpose(0, 1), (24,)),
    ("transpose_merge_front", lambda: base((2, 3, 4)).transpose(0, 1), (6, 4)),
    ("transpose_split_last", lambda: base((2, 3, 4)).transpose(0, 1), (3, 2, 2, 2)),
    ("col_slice_flatten", lambda: base((4, 6))[:, :3], (12,)),
    ("col_slice_split_rows", lambda: base((4, 6))[:, :3], (2, 2, 3)),
    ("row_slice_flatten", lambda: base((4, 6))[1:3], (12,)),
    ("step_slice_flatten", lambda: base((4, 6))[:, ::2], (12,)),
    ("expand_flatten", lambda: base((3,)).expand(4, 3), (12,)),
    ("expand_split", lambda: base((3,)).expand(4, 3), (2, 2, 3)),
    ("t_then_t_back", lambda: base((4, 6)).t(), (4, 6)),
    ("t_flatten", lambda: base((4, 6)).t(), (24,)),
    ("unsqueeze_squeeze", lambda: base((2, 3)).unsqueeze(1), (2, 3)),
]


@pytest.mark.parametrize("name,build,new_shape", VIEW_CASES, ids=[c[0] for c in VIEW_CASES])
def test_stretch_view_possible_matches_torch(name, build, new_shape):
    t = build()
    try:
        t.view(new_shape)
        expected = True
    except RuntimeError:
        expected = False
    assert view_possible(tuple(t.shape), tuple(t.stride()), new_shape) == expected


def test_stretch_view_cases_include_failures():
    outcomes = set()
    for _, build, new_shape in VIEW_CASES:
        try:
            build().view(new_shape)
            outcomes.add(True)
        except RuntimeError:
            outcomes.add(False)
    assert outcomes == {True, False}
