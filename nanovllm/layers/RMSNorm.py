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
        x_fp32 = x.float()
        weight_fp32 = self.weight.float()
        rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        # 应用RMSNorm
        out = weight_fp32 * x_fp32 / rms
        return out.to(x.dtype)