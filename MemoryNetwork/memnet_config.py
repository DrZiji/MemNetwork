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
    batch = 8
    memory_depth = 256
    multi_embed = 8

OGEMemConfig = ObjGuidedEMemConfig()
