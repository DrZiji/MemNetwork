import torch
from torch import nn

"""The base module for memory network"""
"""
根据最近的阅读，我重新整理了关于LSTM和Memory Network的类比过程。
1.LSTM中真正保存长期记忆的结构是cell state c(t),不是hidden state h(t);我不知道LSTM中作者加入hidden state 
的目的是什么，目前看来hidden state完全可以用函数式包裹cell state来代替。
2.也就是说，LSTM中对记忆的更新，其实是对c(t)的更新。
3.之前的理解是将h(t)看作memory,将c(t)看作controller，这个理解需要做出调整。但LSTM仍然可以和Memory Network
类比。

t时刻:
生成写信号w(t)，读信号r(t)。
4.c(t)的写入:input gate 和 forget gate 共同作用，c(t-1)->c(t)，通过w(t)。
5.c(t)的读取:output gate, c(t)->output(t)。
6.LSTM中控制读写信号生成使用feed forward network。

"""
class MemNetBase(nn.Module):
    def __init__(self):
        super(MemNetBase, self).__init__()
    
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
