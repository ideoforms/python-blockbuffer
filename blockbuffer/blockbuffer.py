import numpy as np
from .exceptions import BlockBufferValueException, BlockBufferFullException

BLOCK_BUFFER_DEFAULT_CAPACITY_BLOCKS = 8

class BlockBuffer(object):
    def __init__(self, block_size, hop_size=None, capacity=None):
        """
        Args:
            block_size: The number of samples to return per block.
            hop_size: The amount the read head should be moved forward per block.
            capacity: The total buffer capacity in samples. Defaults to block_size * 8.
        """
        self.block_size = block_size
        self.hop_size = hop_size if hop_size is not None else block_size
        self.write_position = 0
        self.read_position = 0
        self.length = 0

        if capacity:
            self.capacity = capacity
        else:
            self.capacity = self.block_size * BLOCK_BUFFER_DEFAULT_CAPACITY_BLOCKS

        if self.hop_size == 0:
            raise BlockBufferValueException("Hop size must be >0")
        if self.block_size == 0:
            raise BlockBufferValueException("Block size must be >0")
        if self.hop_size > self.block_size:
            raise BlockBufferValueException("Hop size must be <= block_size")
        if self.capacity < self.block_size + self.hop_size:
            raise BlockBufferValueException("Capacity must be >= block_size + hop_size")

        self.queue = np.zeros((self.capacity))
        self.return_buffer = np.zeros((self.block_size))

    def __iter__(self):
        return self.blocks

    def extend(self, frames):
        """
        Append frames to the buffer.
        Safe for usage in real-time audio applications, as no memory allocation or system I/O is done

        Args:
            frames: A 1D array of frames to process.
        """
        if len(np.shape(frames)) > 1:
            raise BlockBufferValueException("Block buffer currently only supports 1D data")

        num_frames = len(frames)
        if self.length + num_frames > self.capacity:
            raise BlockBufferFullException("Block buffer overflowed")

        if self.write_position + num_frames <= self.capacity:
            self.queue[self.write_position:self.write_position+num_frames] = frames
        else:
            remaining_frames = self.capacity - self.write_position
            self.queue[self.write_position:] = frames[:remaining_frames]
            self.queue[:num_frames - remaining_frames] = frames[remaining_frames:]

        self.length += num_frames
        self.write_position = (self.write_position + num_frames) % self.capacity

    def get(self):
        """
        Returns a block of samples from the buffer, if any are available.

        Returns:
            An array of exactly `block_size` samples, or None if no more blocks remain
            to be read.
        """
        if self.length >= self.block_size:
            if self.read_position + self.block_size <= self.capacity:
                rv = self.queue[self.read_position:self.read_position + self.block_size]
            else:
                remaining_frames = self.capacity - self.read_position
                self.return_buffer[:remaining_frames] = self.queue[self.read_position:]
                self.return_buffer[remaining_frames:] = self.queue[:self.block_size - remaining_frames]
                rv = self.return_buffer
            self.length -= self.hop_size
            self.read_position = (self.read_position + self.hop_size) % self.capacity
            return rv
        return None

    @property
    def blocks(self):
        """
        Returns:
            A generator which yields remaining blocks of samples.
        """
        while True:
            rv = self.get()
            if rv is not None:
                yield rv
            else:
                return
