import sys
sys.path.append("d:\\MemNetwork")

import torch
import numpy as np
from MemoryNetwork.ObjGuidedEMem import OGEMem


if __name__ == "__main__":
    OGEMem_ins = OGEMem(16, 16, -1)
    # a1 = np.random.rand(2, 16, 8, 10)
    # a2 = np.random.rand(2, 16, 8, 10)
    # np.save('numpy_debug_pres.npy', a1)
    # np.save('numpy_debug_prev.npy', a2)
    # read = np.load('numpy_debug_pres.npy')
    for time_step in range(10):
        # prev_obj = torch.from_numpy(np.random.rand(2, 6, 4, 8)).float()
        pres_fea = torch.from_numpy(np.load('numpy_debug_pres.npy')).float()
        prev_fea = torch.from_numpy(np.load('numpy_debug_prev.npy')).float()
        prev_bbox = np.array([[0, 1, 3, 3],
                            [1, 2, 5, 8]])
        prev_trk = [2, 0.5]
        mnet_output = OGEMem_ins(prev_fea, prev_bbox, pres_fea, time_step + 1, prev_trk)
    print()