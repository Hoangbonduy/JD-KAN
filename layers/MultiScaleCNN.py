import torch
import torch.nn as nn

class MultiScaleJumpStream(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(MultiScaleJumpStream, self).__init__()
        
        # Bottleneck: giảm kênh trước convolution để giảm params
        self.bottleneck_dim = d_model // 4
        self.down_proj = nn.Conv1d(d_model, self.bottleneck_dim, kernel_size=1)
        
        # k=1: Bắt nhiễu tức thời
        self.conv1 = nn.Conv1d(in_channels=self.bottleneck_dim, out_channels=self.bottleneck_dim, 
                               kernel_size=1, padding=0)
        
        # k=3: Bắt gai ngắn (padding=1 để giữ size)
        self.conv3 = nn.Conv1d(in_channels=self.bottleneck_dim, out_channels=self.bottleneck_dim, 
                               kernel_size=3, padding=1, padding_mode='zeros')
        
        # k=5: Bắt biến động cụm (padding=2 để giữ size)
        self.conv5 = nn.Conv1d(in_channels=self.bottleneck_dim, out_channels=self.bottleneck_dim, 
                               kernel_size=5, padding=2, padding_mode='zeros')

        # Fusion: Concatenate (3 * bottleneck_dim) -> Conv1x1 -> d_model
        self.fusion_conv = nn.Conv1d(in_channels=self.bottleneck_dim * 3, out_channels=d_model, 
                                     kernel_size=1)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # LayerNorm thay vì BatchNorm: không gây train/eval discrepancy
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Input x: [Batch, Seq_Len, Channels]
        residual = x
        
        # Permute sang [Batch, Channels, Seq_Len] cho Conv1d
        x = x.permute(0, 2, 1)
        
        # Bottleneck projection
        x = self.down_proj(x)
        
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        
        # Concatenate
        out = torch.cat([out1, out3, out5], dim=1) 
        
        # Trộn thông tin
        out = self.fusion_conv(out)
        
        # Trả về [Batch, Seq_Len, Channels]
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        
        # Residual connection: giữ lại thông tin gốc
        return out + residual