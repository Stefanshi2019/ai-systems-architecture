"""Tensor memory-layout experiments. Fill in every TODO in plain Python (no torch except where noted).

A tensor is (storage, shape, strides, storage_offset). Element `index` lives at
    storage[storage_offset + sum(index[d] * strides[d])]
Views change only (shape, strides, storage_offset); the storage is shared.

Tests: uv run pytest phase1_foundations/month01_autograd/test_tensor_layout.py -q
"""

import numpy as np
import math 

def contiguous_strides(shape):
    """Strides (in elements) of a freshly allocated row-major tensor of this shape.

    Example: (2, 3, 4) -> (12, 4, 1)
    """
    
    j = 1
    l = []
    for i in reversed(shape):
        l.append(j)
        j = j * i
    return reversed(l) 


def flat_index(index, strides, storage_offset=0):
    """Position in the flat storage of the element at `index` (a tuple, one entry per dim)."""
    return storage_offset + np.dot(index, strides)


def transpose_layout(shape, strides, dim0, dim1):
    """(shape, strides) after swapping dim0 and dim1. No data moves."""
    shape_l = list(shape)
    strides_l = list(strides)
    shape_l[dim0], shape_l[dim1] = shape_l[dim1], shape_l[dim0]
    strides_l[dim0], strides_l[dim1] = strides_l[dim1], strides_l[dim0]
    shape = tuple(shape_l)
    strides = tuple(strides_l)
    return (shape, strides)
    


def slice_layout(shape, strides, storage_offset, dim, start, stop, step=1):
    """(shape, strides, storage_offset) of `t[..., start:stop:step, ...]` along `dim`.

    start >= 0, step >= 1; `stop` may exceed the dimension size (clamp it).
    """
    shape = list(shape)
    strides = list(strides)
    old_strides = list(strides)
    ### should I assume both start and stop are always legal?
    if (stop > shape[dim]):
        stop = shape[dim]
    reduction = math.ceil ((stop - start) / step)
    shape[dim] = reduction
    strides[dim] = strides[dim] * step
    offset = old_strides[dim] * start + storage_offset

    return tuple(shape), tuple(strides), offset


def expand_layout(shape, strides, new_shape):
    """(new_shape, new_strides) of `t.expand(new_shape)`.

    Expanded dimensions (size 1 -> n, and new leading dimensions) must not advance in memory.
    A dimension whose size does not change keeps its stride. `new_shape` has no -1 entries.
    """
    # TODO
    new_stride = tuple(strides)
    if len(shape) < len(new_shape):
        lead = len(new_shape) - len(shape)
        shape = (1,) * lead + tuple(shape)
        new_stride = (0,) * lead + tuple(strides)
    new_stride = list(new_stride)

    for i in range(len(shape)):
        if (shape[i] != new_shape[i]):
            if shape[i] == 1 :
                new_stride [i] = 0
            else:
                raise error
        
    return new_shape, tuple(new_stride)
    raise NotImplementedError


def is_contiguous_layout(shape, strides):
    """True if the layout is row-major dense, i.e. what `t.is_contiguous()` reports.

    Hint: walk from the last dim to the first, tracking the stride a dense layout would need.
    Think about what should happen to dimensions of size 1.
    """
    rstrides = list(strides)
    cstrides = list(contiguous_strides(shape))
    for i in range(0, len(strides) , 1):
        if shape[i] ==1:
            continue
        if strides[i] != cstrides[i]:
            return False


    return True
        
        
    # TODO
    raise NotImplementedError


def shares_storage(a, b):
    """True if torch tensors `a` and `b` are backed by the same memory allocation.

    (This one uses torch: look at `Tensor.untyped_storage()`.)
    """
    # TODO
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

def view_possible(shape, strides, new_shape):
    """True if `t.view(new_shape)` is legal for a tensor with this layout (no copy needed).

    STRETCH. `new_shape` has no -1 entries and the same number of elements as `shape`; no zero-sized dims.
    Think in terms of groups of adjacent dims that are laid out as one dense run
    (stride[i] == stride[i+1] * shape[i+1]); a view may split or merge dims only inside such a group.
    """
       # size-1 dims never move, so ignore them in the old layout
    dims = [(n, s) for n, s in zip(shape, strides) if n != 1]

    # 1. group adjacent old dims into "dense runs"; remember each run's element count
    chunks = []
    for k, (n, s) in enumerate(dims):
        if k > 0 and dims[k - 1][1] == s * n:
            chunks[-1] *= n
        else:
            chunks.append(n)

    # 2. the new shape must split into consecutive groups whose products match the runs
    j = 0
    for chunk in chunks:
        prod = 1
        while j < len(new_shape) and prod < chunk:
            prod *= new_shape[j]
            j += 1
        if prod != chunk:
            return False
    return all(n == 1 for n in new_shape[j:])
