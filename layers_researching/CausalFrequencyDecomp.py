import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalFrequencyDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # Đảm bảo kernel_size là số lẻ để padding đối xứng (nếu cần)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [Batch, Seq, Channel]
        x_in = x.permute(0, 2, 1) # -> [Batch, Channel, Seq]
        
        # --- FIX TỪ GIẢI PHÁP 1 ---
        # Thay vì tạo padding zero thủ công:
        # padding = torch.zeros(...) 
        
        # Ta dùng F.pad với mode='replicate'
        # Padding bên trái (trước) = kernel_size - 1
        # Padding bên phải (sau) = 0 (để đảm bảo tính nhân quả - Causal)
        x_pad = F.pad(x_in, (self.kernel_size - 1, 0), mode='replicate')
        
        # Tính Trend
        x_trend = self.avg(x_pad)
        
        # Tính Residual (Spike)
        x_res = x_in - x_trend
        
        return x_trend.permute(0, 2, 1), x_res.permute(0, 2, 1)