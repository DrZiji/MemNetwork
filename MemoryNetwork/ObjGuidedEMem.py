import torch
import torch.nn as nn
from base import MemNetBase

class OGEMem(MemNetBase):
    """
                |
    ------------|
    
    """


    def __init__(self):
        super(OGEMem, self).__init__()

    def on_step_start(self):
        pass

    def on_generate_write_weight(self):
        pass

    def on_memory_write(self):
        pass

    def on_generate_read_weight(self):
        pass

    def on_memory_read(self):
        pass

    def on_step_end(self):
        pass