import torch
import torch.nn as nn
from .rKANLayer import rKANBlock

class DiffusionLayer(nn.Module):
    """
    Nhánh 1: Continuous Flow Stream
    Mô hình hóa động lực học liên tục (Continuous Dynamics) của chuỗi thời gian.
    
    Nhận x_smooth từ SmoothLayer và học đặc trưng qua rKAN để dự báo.
    """
    def __init__(self, input_dim, seq_len, pred_len, kan_order=3):
        super(DiffusionLayer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # KAN layer for smooth feature extraction
        self.smooth_kan = rKANBlock(input_dim, input_dim, kan_order=kan_order)
        
        # Temporal Projection (Time-Mixing)
        # Chiếu trục thời gian từ Seq_Len -> Pred_Len
        # Đây là bước "Extrapolation" (Ngoại suy)
        self.time_proj = nn.Linear(seq_len, pred_len)
        
        # LayerNorm để ổn định trước khi output
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x_smooth):
        # x_smooth: [Batch, Seq_Len, Channels] - Output từ SmoothLayer
        
        # Bước 1: Học đặc trưng thông qua Smooth KAN
        x_cont = self.smooth_kan(x_smooth)
        x_cont = self.norm(x_cont)
        
        # Bước 2: Ngoại suy tương lai (Projection)
        # Permute để Linear tác động vào chiều Seq_Len
        # [Batch, Seq_Len, Dim] -> [Batch, Dim, Seq_Len]
        x_cont = x_cont.permute(0, 2, 1)
        
        # Chiếu: [Batch, Dim, Seq_Len] -> [Batch, Dim, Pred_Len]
        future_cont = self.time_proj(x_cont)
        
        # Permute lại: [Batch, Pred_Len, Dim]
        return future_cont.permute(0, 2, 1)