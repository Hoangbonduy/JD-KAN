import torch
import torch.nn as nn
from layers.rKAN import rKANBlock  # Từ code bạn

class DilatedJumpLayer(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilations=[1, 2, 4, 8, 16], kan_order=3, dropout=0.3):
        super().__init__()
        self.dilations = dilations  # Tăng cấp số nhân như bạn nói (bắt multi-scale: tức thời đến dài hạn)
        
        # Stack dilated convs (causal, 1D)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, 
                      dilation=d, padding=(kernel_size-1)*d,  # Causal padding: chỉ past
                      padding_mode='zeros') 
            for d in dilations
        ])
        
        # rKAN để học phi tuyến (sau convs)
        self.rkan = rKANBlock(d_model, d_model, kan_order=kan_order, dropout=dropout)
        
        # Norm và activation (từ TCN: ReLU hoặc GELU)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()  # Hoặc ReLU
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, C] → Permute [B, C, L] cho Conv1d
        residual = x  # Residual connection (như TCN/WaveNet)
        seq_len = x.size(1)  # Store original sequence length
        x = x.permute(0, 2, 1)
        
        # Áp dụng từng dilated conv (multi-scale)
        for conv in self.convs:
            x = conv(x)  # Dilated conv
            # Trim to original length (causal: keep left, remove right padding)
            x = x[:, :, :seq_len]
            x = self.act(x)
            x = self.dropout(x)
        
        # Trở về [B, L, C]
        x = x.permute(0, 2, 1)
        
        # rKAN để mix channels phi tuyến (bắt amplitude/phase của jumps)
        x = self.rkan(x)
        
        # Norm và residual
        x = self.norm(x + residual)  # Giữ info gốc, tránh vanishing gradient
        
        return x