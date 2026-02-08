import torch
import torch.nn as nn
import torch.nn.functional as F

class LightCrossAttention(nn.Module):
    """
    Lightweight Cross-Attention mechanism.
    Không dùng MultiheadAttention của PyTorch để tránh lỗi chia hết số heads 
    khi số lượng features (channels) nhỏ hoặc lẻ.
    
    Q: Continuous Stream (Trend)
    K, V: Jump Stream (Events)
    """
    def __init__(self, d_model, dropout=0.1):
        super(LightCrossAttention, self).__init__()
        self.scale = d_model ** -0.5
        
        # Projection layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_q, x_kv):
        # x_q (Continuous): [Batch, Len, Dim]
        # x_kv (Jump):      [Batch, Len, Dim]
        
        residual = x_q
        
        # 1. Linear Projections
        q = self.w_q(x_q)
        k = self.w_k(x_kv)
        v = self.w_v(x_kv)
        
        # 2. Attention Score: Q * K^T
        # [Batch, Len, Dim] x [Batch, Dim, Len] -> [Batch, Len, Len]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 3. Context: Score * V
        # [Batch, Len, Len] x [Batch, Len, Dim] -> [Batch, Len, Dim]
        context = torch.matmul(attn, v)
        
        # 4. Output Projection & Residual
        output = self.out_proj(context)
        return self.norm(residual + output)

class FusionLayer(nn.Module):
    """
    Hợp nhất 2 nhánh dự báo.
    Công thức: Y_final = Y_cont_adjusted + (1 + alpha) * Y_jump
    """
    def __init__(self, d_model, dropout=0.1):
        super(FusionLayer, self).__init__()
        
        # 1. Cross Interaction
        # Để Nhánh 1 (Cont) biết về sự tồn tại của Nhánh 2 (Jump) và điều chỉnh
        self.interaction = LightCrossAttention(d_model, dropout)
        
        # 2. Learnable Mixing Parameter (Alpha)
        # alpha học riêng cho từng channel (feature)
        # Khởi tạo = 0 để ban đầu factor = 1 (cộng bình thường)
        # Shape: [1, 1, d_model] để broadcasting
        self.alpha = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 3. Final Smoothing (Optional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_cont, x_jump):
        # x_cont: [Batch, Pred_Len, Channels]
        # x_jump: [Batch, Pred_Len, Channels]
        
        # Bước 1: Interaction
        # Continuous stream được điều chỉnh bởi Jump stream
        # (Ví dụ: Nếu Jump báo hiệu cú sốc, Cont stream có thể giảm độ trễ)
        x_cont_adjusted = self.interaction(x_q=x_cont, x_kv=x_jump)
        
        # Bước 2: Adaptive Mixing
        # Tính factor jump impact: (1 + tanh(alpha))
        # tanh giữ alpha trong khoảng [-1, 1], giúp training ổn định hơn
        jump_impact = 1.0 + torch.tanh(self.alpha)
        
        # Tổng hợp
        y_final = x_cont_adjusted + (jump_impact * x_jump)
        
        return self.dropout(y_final)