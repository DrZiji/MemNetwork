import torch
import numpy
from torch import nn
from base import MemNetBase
from MemTrackConfig import MTConfig

class MemTrack(MemNetBase):
    """
    MemTrack整体从t-1到t时间步各信号生成和传递过程。

                        |       t
    ---------------------------------------------------------------
    on_step_start:      |   None
    on_memory_write:    |   M(t-1)->M(t),via W(t-1)
                        |   input frame t, generate C(t), R(t)
    on_memory_read:     |   M(t)->O(t)
    on_step_end:        |   W(t), A(t), U(t), G(t)
    """

    def __init__(self, gpu_id):
        super(MemTrack, self).__init__()
        if gpu_id < 0:
            self.device = 'cpu'
        else:
            self.device = 'cuda:'+str(gpu_id)


    def on_step_start(self, time_step):
        """
        :param time_step: Memory Network和LSTM类似，样本是时间序列，该变量记录本次调用的时间步。值从1开始
        """
        assert time_step > 0, "wrong time_step, must bigger than 1"
        if time_step == 1:
            # 初始化变量
            with torch.cuda.device(torch.device(self.device)):
                init_memory = torch.zeros(MTConfig.memory_size, dtype=torch.float)
                # init_write, init_usage, init_gate one-hot
                temp1 = torch.zeros((MTConfig.batch, MTConfig.memory_slot), dtype=torch.float)
                index1 = torch.zeros((MTConfig.batch, 1), dtype=torch.long)
                init_write = temp1.scatter(1, index1, 1.0)
                init_usage = temp1.scatter(1, index1, 1.0)
                
                # init_gate, one-hot
                temp2 = torch.zeros((MTConfig.batch, MTConfig.gate_num), dtype=torch.float)
                index2 = torch.ones((MTConfig.batch, 1), dtype=torch.long) * (MTConfig.gate_num - 1)
                init_gate = temp2.scatter(1, index2, 1.0)
                init_decay = torch.zeros((MTConfig.batch, 1), dtype=torch.float)
            
            init_erase = init_decay * init_gate[:, 1].unsqueeze(-1) + init_gate[:, 2].unsqueeze(-1)
            return init_memory, init_write, init_usage, init_erase
        else:
            return

    def on_memory_write(self, *prev_states):
        "track, write, erase, memory in t-1 state"
        prev_track, prev_w, prev_e, prev_m = prev_states
        pass
    
    def on_memory_read(self, read):
        pass

    def on_step_end(self, prev_usage, read):
        """
        t时刻结束，生成t时刻的其他信号，其中，write需要传到下一个时刻。
        """
        allocation = self.on_generate_allocation_weight(prev_usage)
        write = self.on_generate_write_weight(allocation, read)
        usage = self.on_generate_usage_weight(prev_usage, write, read)
        return write

    def on_generate_controller_state(self, input):
        pass

    def on_generate_write_weight(self, a_weight, r_weight):
        write = None
        return write

    def on_generate_allocation_weight(self, prev_u_weight):
        allocation = None
        return allocation
    
    def on_generate_read_weight(self):
        read = None
        return read

    def on_generate_usage_weight(self, prev_u_weight, w_weight, r_weight):
        usage = None
        return usage
    
