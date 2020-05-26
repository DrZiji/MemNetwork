class MemTrackConfig:
    batch = 8
    memory_slot = 8
    memory_channel = 256
    slot_width = 6
    slot_height = 6
    memory_size = (batch, memory_slot, memory_channel, slot_height, slot_width)

    gate_num = 3

MTConfig = MemTrackConfig()

class ObjGuidedEMemConfig:
    batch = 2
    memory_depth = 16 # memory slot中的数据channel
    memory_att_depth = 64 # 进行attention操作时的数据维度
    multi_embed = 8 # 将memory映射至不同用途时embedding的个数
    alpha = 0.47
    taw = 0.9

OGEMConfig = ObjGuidedEMemConfig()
