import pytest
import numpy as np
from blockbuffer import BlockBuffer
from blockbuffer import BlockBufferFullException, BlockBufferValueException

def test_blockbuffer_basic():
    bb = BlockBuffer(4)
    assert bb.get() is None
    bb.extend([ 1, 2, 3, 4, 5, 6, 7, 8 ])
    assert np.array_equal(bb.get(), [ 1, 2, 3, 4 ])
    assert np.array_equal(bb.get(), [ 5, 6, 7, 8 ])
    assert bb.get() is None

def test_blockbuffer_capacity():
    bb = BlockBuffer(4, capacity=8)
    bb.extend([ 1, 2, 3, 4 ])
    bb.extend([ 5, 6, 7, 8 ])
    with pytest.raises(BlockBufferFullException):
        bb.extend([ 9 ])

def test_blockbuffer_hop():
    bb = BlockBuffer(8, 2)
    bb.extend([ 1, 2, 3, 4 ])
    assert bb.get() is None
    bb.extend([ 5, 6, 7, 8 ])
    assert np.array_equal(bb.get(), [ 1, 2, 3, 4, 5, 6, 7, 8 ])
    bb.extend([ 9, 10, 11, 12 ])
    assert np.array_equal(bb.get(), [ 3, 4, 5, 6, 7, 8, 9, 10 ])
    assert np.array_equal(bb.get(), [ 5, 6, 7, 8, 9, 10, 11, 12 ])
    assert bb.get() is None

def test_blockbuffer_hop_odd_capacity():
    bb = BlockBuffer(4, 2, capacity=6)
    assert bb.get() is None
    bb.extend([ 1, 2, 3, 4 ])
    assert np.array_equal(bb.get(), [ 1, 2, 3, 4 ])
    assert bb.get() is None
    bb.extend([5, 6, 7, 8])
    assert np.array_equal(bb.get(), [ 3, 4, 5, 6 ])
    assert np.array_equal(bb.get(), [ 5, 6, 7, 8 ])
    assert bb.get() is None

def test_blockbuffer_iterator():
    bb = BlockBuffer(4, 2)
    bb.extend([ 1, 2, 3, 4, 5, 6, 7, 8 ])
    blocks = list(bb)
    assert len(blocks) == 3
    assert np.array_equal(blocks[0], [ 1, 2, 3, 4 ])
    assert np.array_equal(blocks[1], [ 3, 4, 5, 6 ])
    assert np.array_equal(blocks[2], [ 5, 6, 7, 8 ])

def test_blockbuffer_bad_values():
    # invalid hop size
    with pytest.raises(BlockBufferValueException):
        bb = BlockBuffer(4, 0)

    # invalid block size
    with pytest.raises(BlockBufferValueException):
        bb = BlockBuffer(0, 4)

    # invalid hop size vs block size
    with pytest.raises(BlockBufferValueException):
        bb = BlockBuffer(4, 5)

    # invalid capacity (must be >= block_size + hop_size)
    with pytest.raises(BlockBufferValueException):
        bb = BlockBuffer(4, 4, 4)

    # invalid data format (must be 1D)
    bb = BlockBuffer(4)
    with pytest.raises(BlockBufferValueException):
        bb.extend(np.array([ [1, 2], [3, 4]]))