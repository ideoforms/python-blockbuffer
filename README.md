# Python: Buffer audio samples into fixed-sized blocks, with overlap

This small utility package encapsulates a single-consumer, single-producer ringbuffer. 

* Populate the buffer with arbitrary-length arrays
* Query the buffer, and it returns blocks of a specified fixed length, optionally with overlap between successive blocks

It is designed primarily for applying the short-time Fourier transform (STFT) to successive blocks of an input audio stream (see below for example).

It is safe for usage in real-time audio applications, as no memory allocation or system I/O is done within the `extend` method as long as `auto_resize=False` is specified when initialising.

## Usage

To do block-sized buffering with overlap in conjunction with [sounddevice](https://python-sounddevice.readthedocs.io/):

```python
import sounddevice as sd
import numpy as np
import blockbuffer

block_size = 1024
hop_size = 128

bb = blockbuffer.BlockBuffer(block_size=block_size,
                             hop_size=hop_size,
                             num_channels=2)

def input_callback(data, num_frames, time, status):
    global bb
    bb.extend(data)
    for block in bb:
        block_windowed = block.T * np.hanning(block_size)
        block_spectrum = np.fft.rfft(block_windowed)

stream = sd.InputStream(callback=input_callback, channels=1)
stream.start()
```

## Source code

Source code is available on [GitHub](https://github.com/ideoforms/python-blockbuffer).
