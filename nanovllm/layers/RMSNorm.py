import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, dtype, eps=1e-6):
        super().__init__()
        # 这里的hidden_size需要和模型的hidden_size保持一致
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        # 计算RMS
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 应用RMSNorm
        return self.weight * x / rms