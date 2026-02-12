import torch
import torch.nn as nn
from layers.rKAN import rKANBlock
import torch.fft  # Thêm import này

class DilatedJumpLayer(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilations=[1, 2, 4, 8],  # Giảm dilations để tránh over-smooth
                 kan_order=3, dropout=0.1, iterations=2):  # Thêm iterations
        super().__init__()
        self.dilations = dilations
        self.iterations = iterations
        
        # Conv cho smooth stream (giữ nguyên nhưng padding 'replicate' tốt rồi)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, 
                      dilation=d, padding=(kernel_size-1)*d // 2,  # Causal padding symmetric
                      padding_mode='replicate') 
            for d in dilations
        ])
        
        # Fusion layer giữ nguyên
        self.fusion_layer = nn.Conv1d(d_model * len(dilations), d_model, kernel_size=1)
        
        # Thêm refine_conv cho iterative residuals (conv nhẹ để refine spike)
        self.refine_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        # rKAN giữ nguyên
        self.rkan = rKANBlock(d_model, d_model, kan_order=kan_order, dropout=dropout)
        
        # Norm & Act
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x  # [B, L, C]
        x_in = x.permute(0, 2, 1)  # [B, C, L]
        seq_len = x_in.size(2)
        
        # --- THÊM: Seasonal Decomposition (FFT-based) ---
        # Tách low-freq (trend + seasonal) vào smooth, high-freq vào spike init
        fft_x = torch.fft.rfft(x_in, dim=2)
        freq_cutoff = seq_len // 10  # Cutoff empirical, adjust dựa data (e.g., 1/10 freq cho seasonal)
        low_freq = torch.zeros_like(fft_x)
        low_freq[:, :, :freq_cutoff] = fft_x[:, :, :freq_cutoff]
        x_seasonal = torch.fft.irfft(low_freq, dim=2)[:, :, :seq_len]
        x_high_freq = x_in - x_seasonal  # High-freq init cho spike
        
        # --- Iterative Decomposition ---
        x_smooth = x_seasonal  # Init smooth với seasonal
        x_spike = x_high_freq
        
        for _ in range(self.iterations):
            # Smooth stream: Multi-dilated convs trên current x_smooth
            jump_outputs = []
            for conv in self.convs:
                out = conv(x_smooth)
                out = out[:, :, :seq_len]  # Trim
                out = self.act(out)
                out = self.dropout(out)
                jump_outputs.append(out)
            
            x_concat = torch.cat(jump_outputs, dim=1)
            x_smooth = self.fusion_layer(x_concat)  # Update smooth
            
            # Update spike: Original - smooth, rồi refine bằng conv nhẹ
            x_spike = x_in - x_smooth
            x_spike = self.refine_conv(x_spike)  # Refine residuals
            
        # --- Dual-Input Learning ---
        x_smooth_t = x_smooth.permute(0, 2, 1)
        x_spike_t = x_spike.permute(0, 2, 1)
        
        base_out = self.rkan.forward_base(x_smooth_t)
        kan_out = self.rkan.forward_kan(x_spike_t)
        
        x_out = base_out + kan_out
        
        # Residual + Norm
        x_out = self.norm(x_out + residual)
        
        return x_out