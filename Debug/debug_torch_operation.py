import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def repeat_debug(A: torch.Tensor, B: torch.Tensor):
    # A.shape [b, l_A, c];B.shape [b, l_b, c]

    # A = torch.tensor([[[1.], [2.], [3.], [4.]]])  # [1, 4, 1]
    # B = torch.tensor([[[1.], [2.], [3.]]])  # [1, 3, 1]
    # C = repeat_debug(A, B)
    b, l_A, c = A.shape
    _, l_B, _ = B.shape

    A = A.unsqueeze(2)
    A_expand = A.repeat(1, 1, l_B, 1).view(-1, c)
    B = B.unsqueeze(1)
    B_expand = B.repeat(1, l_A, 1, 1).view(-1, c)

    cos_simi_mat = F.cosine_similarity(A_expand, B_expand).view(b, l_A, l_B)
    return cos_simi_mat


if __name__ == "__main__":
    # A = torch.from_numpy(np.random.rand(2, 4, 4))
    # A = F.softmax(A, dim=2)
    # A = torch.from_numpy(np.random.rand(2, 4, 4))
    # B = A[0, (1, 3), :]
    # print()
    A = torch.arange(127, -1, -1)
    print()