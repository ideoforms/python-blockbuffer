class BlockBufferFullException (Exception):
    """ Block buffer has exceeded capacity. """
    pass

class BlockBufferValueException (Exception):
    """ Invalid argument passed to block buffer. """
    pass