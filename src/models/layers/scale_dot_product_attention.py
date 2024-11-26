
import math
import time
from torch import nn
import torch

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e9):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = torch.matmul(q, k_t) / math.sqrt(d_tensor)  # scaled dot product
        # 2. apply masking (opt)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        # 4. multiply with Value
        v = torch.matmul(score,v)

        return v, score
