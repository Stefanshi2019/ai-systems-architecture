"""Tensor memory-layout experiments. Fill in every TODO in plain Python (no torch except where noted).

A tensor is (storage, shape, strides, storage_offset). Element `index` lives at
    storage[storage_offset + sum(index[d] * strides[d])]
Views change only (shape, strides, storage_offset); the storage is shared.

Tests: uv run pytest phase1_foundations/month01_autograd/test_tensor_layout.py -q
"""


def contiguous_strides(shape):
    """Strides (in elements) of a freshly allocated row-major tensor of this shape.

    Example: (2, 3, 4) -> (12, 4, 1)
    """
    # TODO
    raise NotImplementedError


def flat_index(index, strides, storage_offset=0):
    """Position in the flat storage of the element at `index` (a tuple, one entry per dim)."""
    # TODO
    raise NotImplementedError


def transpose_layout(shape, strides, dim0, dim1):
    """(shape, strides) after swapping dim0 and dim1. No data moves."""
    # TODO
    raise NotImplementedError


def slice_layout(shape, strides, storage_offset, dim, start, stop, step=1):
    """(shape, strides, storage_offset) of `t[..., start:stop:step, ...]` along `dim`.

    start >= 0, step >= 1; `stop` may exceed the dimension size (clamp it).
    """
    # TODO
    raise NotImplementedError


def expand_layout(shape, strides, new_shape):
    """(new_shape, new_strides) of `t.expand(new_shape)`.

    Expanded dimensions (size 1 -> n, and new leading dimensions) must not advance in memory.
    A dimension whose size does not change keeps its stride. `new_shape` has no -1 entries.
    """
    # TODO
    raise NotImplementedError


def is_contiguous_layout(shape, strides):
    """True if the layout is row-major dense, i.e. what `t.is_contiguous()` reports.

    Hint: walk from the last dim to the first, tracking the stride a dense layout would need.
    Think about what should happen to dimensions of size 1.
    """
    # TODO
    raise NotImplementedError


def shares_storage(a, b):
    """True if torch tensors `a` and `b` are backed by the same memory allocation.

    (This one uses torch: look at `Tensor.untyped_storage()`.)
    """
    # TODO
    raise NotImplementedError


def view_possible(shape, strides, new_shape):
    """True if `t.view(new_shape)` is legal for a tensor with this layout (no copy needed).

    STRETCH. `new_shape` has no -1 entries and the same number of elements as `shape`; no zero-sized dims.
    Think in terms of groups of adjacent dims that are laid out as one dense run
    (stride[i] == stride[i+1] * shape[i+1]); a view may split or merge dims only inside such a group.
    """
    # TODO
    raise NotImplementedError
