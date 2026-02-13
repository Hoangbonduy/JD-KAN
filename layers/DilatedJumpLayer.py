import torch
import torch.nn as nn
from layers.AdaptiveWaveletKAN import AdaptiveWaveletKANBlock
import torch.fft

class DilatedJumpLayer(nn.Module):
    def __init__(self, d_model, seq_len, kernel_size=3, dilations=[1, 2, 4, 8],
                 kan_order=3, dropout=0.1, iterations=2, wavelet_type='mexican_hat'): # Thêm wavelet_type
        super().__init__()
        self.dilations = dilations
        self.iterations = iterations
        
        # Conv cho smooth stream (giữ nguyên)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, 
                      dilation=d, padding=(kernel_size-1)*d // 2,
                      padding_mode='replicate') 
            for d in dilations
        ])
        
        self.fusion_layer = nn.Conv1d(d_model * len(dilations), d_model, kernel_size=1)
        self.refine_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        # --- THAY THẾ: rKAN bằng WaveletKANBlock ---
        # Theo báo cáo: Wavelet KAN xử lý tốt các đột biến (spikes) mà Jacobi thất bại [cite: 180]
        self.wav_kan = AdaptiveWaveletKANBlock(
            d_model, d_model, 
            seq_len=seq_len,  # Quan trọng
            dropout=dropout,
            num_wavelets=4    # K=4 như trong ảnh minh họa
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_in = x.permute(0, 2, 1)  # [B, C, L]
        seq_len = x_in.size(2)
        
        # --- Seasonal Decomposition (FFT-based) ---
        fft_x = torch.fft.rfft(x_in, dim=2)
        freq_cutoff = seq_len // 10 
        low_freq = torch.zeros_like(fft_x)
        low_freq[:, :, :freq_cutoff] = fft_x[:, :, :freq_cutoff]
        x_seasonal = torch.fft.irfft(low_freq, dim=2)[:, :, :seq_len]
        x_high_freq = x_in - x_seasonal
        
        # --- Iterative Decomposition ---
        x_smooth = x_seasonal
        x_spike = x_high_freq
        
        for _ in range(self.iterations):
            jump_outputs = []
            for conv in self.convs:
                out = conv(x_smooth)
                out = out[:, :, :seq_len]
                out = self.act(out)
                out = self.dropout(out)
                jump_outputs.append(out)
            
            x_concat = torch.cat(jump_outputs, dim=1)
            x_smooth = self.fusion_layer(x_concat)
            
            x_spike = x_in - x_smooth
            x_spike = self.refine_conv(x_spike)
            
        # --- Dual-Input Learning ---
        x_smooth_t = x_smooth.permute(0, 2, 1) # [B, L, C]
        x_spike_t = x_spike.permute(0, 2, 1)   # [B, L, C]
        
        # Sử dụng WaveletKANBlock
        # Nhánh Base học trên phần Smooth (Low frequency)
        base_out = self.wav_kan.forward_base(x_smooth_t)
        
        # Nhánh KAN (Wavelet) học trên phần Spike (High frequency)
        # Wavelet rất nhạy với các biến động nhanh trong x_spike_t [cite: 146, 147]
        kan_out = self.wav_kan.forward_kan(x_spike_t)
        
        # Mix kết quả
        # Lưu ý: Báo cáo gợi ý dùng Concatenation thay vì phép cộng (base + kan) nếu được
        # Nhưng để giữ nguyên kích thước tensor, ta dùng cơ chế mix có sẵn của block
        x_out = self.wav_kan._mix(base_out, kan_out)
        
        x_out = self.norm(x_out + residual)
        
        return x_out