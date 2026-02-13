import torch
import torch.nn as nn

class CausalFrequencyDecomp(nn.Module):
    """
    Thay thế FrequencyDecomp cũ.
    Dùng AvgPool với Padding bên trái để đảm bảo tính nhân quả (Causal),
    không dùng FFT để tránh méo tín hiệu và hiện tượng Gibbs.
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel] -> permute -> [B, C, L]
        x_in = x.permute(0, 2, 1)
        
        # Causal Padding: Chỉ thêm số 0 vào phía trước (quá khứ)
        padding = torch.zeros((x_in.shape[0], x_in.shape[1], self.kernel_size - 1), device=x.device)
        x_pad = torch.cat([padding, x_in], dim=-1)
        
        # 1. Tách Trend (Low Frequency)
        x_trend = self.avg(x_pad)
        
        # 2. Tách Residual/Spike (High Frequency)
        # Phép trừ trực tiếp giữ nguyên độ nhọn của gai
        x_res = x_in - x_trend
        
        return x_trend.permute(0, 2, 1), x_res.permute(0, 2, 1)