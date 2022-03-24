import numpy as np
from .exceptions import BlockBufferValueException, BlockBufferFullException

BLOCK_BUFFER_DEFAULT_CAPACITY_BLOCKS = 64

class BlockBuffer(object):
    def __init__(self, block_size, hop_size=None, num_channels=1, capacity=None, auto_resize=False):
        """
        Args:
            block_size: The number of samples to return per block.
            hop_size: The amount the read head should be moved forward per block.
            num_channels: The number of channels to allocate.
            capacity: The total buffer capacity in samples. Defaults to block_size * 8.
            auto_resize: Automatically resize the buffer if it is extended beyond capacity.
                         Does memory allocation, so should not be used in real-time threads.
        """
        self.block_size = block_size
        self.hop_size = hop_size if hop_size is not None else block_size
        self.num_channels = num_channels
        self.write_position = 0
        self.read_position = 0
        self.length = 0
        self.auto_resize = auto_resize

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

        #--------------------------------------------------------------------------------
        # Ringbuffer to store the entire audio queue.
        #--------------------------------------------------------------------------------
        self.queue = np.zeros((self.capacity, self.num_channels))

        #--------------------------------------------------------------------------------
        # Used to store and return each buffer of audio.
        #--------------------------------------------------------------------------------
        self.return_buffer = np.zeros((self.block_size, self.num_channels))

    def __iter__(self):
        return self.blocks

    def extend(self, frames):
        """
        Append frames to the buffer.
        Safe for usage in real-time audio applications, as no memory allocation or system I/O is done

        Args:
            frames: An array of frames to process. Can be a 1D or 2D array, either Python
                    native list or numpy.ndarray.
        """

        #--------------------------------------------------------------------------------
        # Type checking and array validation.
        #--------------------------------------------------------------------------------
        if type(frames) == np.ndarray:
            if frames.ndim > 2:
                raise BlockBufferValueException("Invalid number of dimensions in frames")
            elif (frames.ndim == 2 and frames.shape[1] != self.num_channels) or \
                    (frames.ndim == 1 and self.num_channels > 1):
                raise BlockBufferValueException("Invalid number of channels in frames (expected %d, got %d)" %
                                 (self.num_channels, frames.shape[1]))
        else:
            if self.num_channels == 1:
                if not isinstance(frames[0], (int, float)):
                    raise BlockBufferValueException("Invalid number of dimensions in frames")
            else:
                if type(frames[0]) != list:
                    raise BlockBufferValueException("Invalid number of dimensions in frames")
                if len(frames[0]) != self.num_channels:
                    raise BlockBufferValueException("Invalid number of channels in frames (expected %d, got %d)" %
                                     (self.num_channels, len(frames[0])))
        num_frames = len(frames)

        #--------------------------------------------------------------------------------
        # Resize the buffer (if enabled)
        #--------------------------------------------------------------------------------
        if self.length + num_frames > self.capacity:
            if self.auto_resize:
                size_increase = self.length + num_frames - self.capacity
                self.queue = np.pad(self.queue, ((0, size_increase), (0, 0)))
                self.capacity += size_increase
            else:
                raise BlockBufferFullException("Block buffer overflowed")

        #--------------------------------------------------------------------------------
        # Write the samples.
        # Logic is complex due to having to jump through hoops to avoid memory
        # allocations.
        #--------------------------------------------------------------------------------
        self.write_position = self.write_position % self.capacity

        if self.write_position + num_frames <= self.capacity:
            if self.num_channels == 1:
                self.queue[self.write_position:self.write_position + num_frames, 0] = frames
            elif type(frames) == np.ndarray:
                self.queue[self.write_position:self.write_position + num_frames] = frames
            else:
                for index, row in enumerate(frames):
                    self.queue[self.write_position + index] = row
        else:
            remaining_frames = self.capacity - self.write_position
            if self.num_channels == 1:
                self.queue[self.write_position:, 0] = frames[:remaining_frames]
                self.queue[:num_frames - remaining_frames, 0] = frames[remaining_frames:]
            elif type(frames) == np.ndarray:
                self.queue[self.write_position:] = frames[:remaining_frames]
                self.queue[:num_frames - remaining_frames] = frames[remaining_frames:]
            else:
                for index, row in enumerate(frames):
                    self.queue[(self.write_position + index) % self.capacity] = row

        #--------------------------------------------------------------------------------
        # Update length and write position
        #--------------------------------------------------------------------------------
        self.length += num_frames
        self.write_position = self.write_position + num_frames

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
            if self.num_channels == 1:
                return rv[:, 0]
            else:
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
