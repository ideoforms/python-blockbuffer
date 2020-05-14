"""

BlockBuffer
~~~~~~
Buffer audio samples into fixed-sized blocks, with overlap.


Usage
~~~~~
import blockbuffer
bb = blockbuffer.BlockBuffer(block_size=1024,
                             hop_size=128)
bb.extend(array_of_samples)
for block in bb.blocks:
    assert len(block) == 1024
"""

from .blockbuffer import BlockBuffer
from .exceptions import BlockBufferFullException, BlockBufferValueException

__author__ = "Daniel Jones <http://www.erase.net/>"
__all__ = [
    "BlockBuffer",
    "BlockBufferFullException",
    "BlockBufferValueException"
]
