import torch
import torch.nn as nn
from .rKANLayer import rKANBlock

class DiffusionLayer(nn.Module):
    """
    Nhánh 1: Continuous Flow Stream
    Mô hình hóa động lực học liên tục (Continuous Dynamics) của chuỗi thời gian.
    
    Cải tiến: Áp dụng Multi-order rKAN.
    - Trend: Dùng rKAN bậc thấp (Low Order) -> Smoothness priority.
    - Season: Dùng rKAN bậc cao (High Order) -> Fitting priority.
    """
    def __init__(self, input_dim, seq_len, pred_len, kan_order=3):
        super(DiffusionLayer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Trend Modeling (Low-Frequency)
        # Ép buộc bậc thấp (vd: 2 hoặc 3) để tránh Overfitting vào nhiễu
        trend_order = max(1, int(kan_order / 2) + 1) 
        self.trend_kan = rKANBlock(input_dim, input_dim, kan_order=trend_order)
        
        # 2. Seasonality Modeling (Mid-Frequency)
        # Dùng bậc đầy đủ như cấu hình để bắt dao động phức tạp
        self.season_kan = rKANBlock(input_dim, input_dim, kan_order=kan_order)
        
        # 3. Temporal Projection (Time-Mixing)
        # Chiếu trục thời gian từ Seq_Len -> Pred_Len
        # Đây là bước "Extrapolation" (Ngoại suy)
        self.time_proj = nn.Linear(seq_len, pred_len)
        
        # LayerNorm để ổn định trước khi output
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x_trend, x_season):
        # x_trend: [Batch, Seq_Len, Dim]
        # x_season: [Batch, Seq_Len, Dim]
        
        # Bước 1: Học đặc trưng động lực học (Dynamics Learning)
        # Xử lý song song: Trend học cái "chậm", Season học cái "vừa"
        trend_feat = self.trend_kan(x_trend)
        season_feat = self.season_kan(x_season)
        
        # Hợp nhất Continuous Flow
        x_cont = trend_feat + season_feat
        x_cont = self.norm(x_cont)
        
        # Bước 2: Ngoại suy tương lai (Projection)
        # Permute để Linear tác động vào chiều Seq_Len
        # [Batch, Seq_Len, Dim] -> [Batch, Dim, Seq_Len]
        x_cont = x_cont.permute(0, 2, 1)
        
        # Chiếu: [Batch, Dim, Seq_Len] -> [Batch, Dim, Pred_Len]
        future_cont = self.time_proj(x_cont)
        
        # Permute lại: [Batch, Pred_Len, Dim]
        return future_cont.permute(0, 2, 1)