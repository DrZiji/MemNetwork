import sys
sys.path.append("d:\\MemNetwork")

import torch
import numpy as np
from MemoryNetwork.ObjGuidedEMem import OGEMem


if __name__ == "__main__":
    OGEMem_ins = OGEMem(8, 8, -1)

    prev_obj = torch.from_numpy(np.random.rand(2, 4, 4, 8)).float()
    pres_fea = torch.from_numpy(np.random.rand(2, 8, 8, 8)).float()
    prev_fea = torch.from_numpy(np.random.rand(2, 8, 8, 8)).float()
    prev_trk = [2, 0.5]
    for time_step in range(2):
        mnet_output = OGEMem_ins(prev_fea, None, prev_trk, pres_fea, time_step + 1, prev_obj)
    print()