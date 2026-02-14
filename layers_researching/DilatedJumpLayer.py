import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_researching.AdaptiveWaveletKAN import AdaptiveWaveletKANBlock
import torch.fft

class DilatedJumpLayer(nn.Module):
    def __init__(self, d_model, seq_len, kernel_size=3, dilations=[1, 2, 4, 8],
                 kan_order=3, dropout=0.1, iterations=2, wavelet_type='mexican_hat'):
        super().__init__()
        self.dilations = dilations
        self.iterations = iterations
        self.seq_len = seq_len
        # Giả định số lượng wavelet con (num_wavelets) là 4 giống như trong các file khác
        # Nếu muốn cấu hình linh hoạt, bạn có thể thêm tham số num_wavelets vào __init__
        self.num_wavelets = 4 
        
        # --- 1. DILATED CONVOLUTIONS (Cơ chế nhìn xa) ---
        # padding_mode='replicate' -> SỬA LỖI VÙNG 0-10 (Cold Start)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, 
                      dilation=d, padding=(kernel_size-1)*d // 2,
                      padding_mode='replicate') 
            for d in dilations
        ])
        
        self.fusion_layer = nn.Conv1d(d_model * len(dilations), d_model, kernel_size=1)
        self.refine_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='replicate')
        
        # --- 2. HYPERNETWORK CNN-1D (Cơ chế thích ứng pha) ---
        # Thay vì Linear, dùng CNN để bắt độ dốc (Slope) của xu hướng
        # SỬA LỖI TRỄ PHA (Phase Lag)
        self.hypernet_cnn = nn.Sequential(
            # Conv 1: Phát hiện xu hướng tăng/giảm cục bộ
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            
            # Conv 2: Tổng hợp ngữ cảnh
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.SiLU(),
            
            # Global Average Pooling: Đúc kết thành vector tham số
            nn.AdaptiveAvgPool1d(1) 
        )
        
        # Chiếu ra tham số a (scale) và b (translation) cho Wavelet
        self.hypernet_proj = nn.Linear(d_model // 2, self.num_wavelets * 2)
        
        # --- 3. AMPLITUDE GATING (Cơ chế nhân) ---
        # Học cách khuếch đại Spike dựa trên Trend
        # SỬA LỖI BIÊN ĐỘ THẤP (Undershooting)
        self.amplitude_gate = nn.Linear(d_model, d_model)

        # --- 4. WAVELET KAN BLOCK ---
        self.wav_kan = AdaptiveWaveletKANBlock(
            d_model, d_model, 
            seq_len=seq_len,
            dropout=dropout,
            num_wavelets=self.num_wavelets
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_in = x.permute(0, 2, 1)  # [Batch, Channel, Seq_Len]
        seq_len = x_in.size(2)
        
        # --- A. FFT DECOMPOSITION (Tách Tần số) ---
        fft_x = torch.fft.rfft(x_in, dim=2)
        freq_cutoff = seq_len // 10 
        low_freq = torch.zeros_like(fft_x)
        low_freq[:, :, :freq_cutoff] = fft_x[:, :, :freq_cutoff]
        x_seasonal = torch.fft.irfft(low_freq, dim=2)[:, :, :seq_len]
        x_high_freq = x_in - x_seasonal
        
        # --- B. ITERATIVE REFINEMENT ---
        x_smooth = x_seasonal # [B, C, L] - Trend
        x_spike = x_high_freq # [B, C, L] - Detail
        
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
            
        # Chuẩn bị input cho KAN
        x_smooth_t = x_smooth.permute(0, 2, 1) # [B, L, C]
        x_spike_t = x_spike.permute(0, 2, 1)   # [B, L, C]
        
        # --- C. DYNAMIC PARAMETER GENERATION (CNN-Based) ---
        # Input cho CNN phải là [B, C, L], dùng ngay x_smooth (chưa permute)
        # x_smooth chứa thông tin xu hướng sạch nhất để điều khiển pha
        
        # 1. Trích xuất đặc trưng ngữ cảnh (Context)
        cnn_feat = self.hypernet_cnn(x_smooth) # -> [B, C//2, 1]
        cnn_feat = cnn_feat.squeeze(-1)        # -> [B, C//2]
        
        # 2. Sinh tham số động
        params = self.hypernet_proj(cnn_feat)  # -> [B, num_wavelets * 2]
        params = params.view(-1, 1, self.num_wavelets, 2) # Reshape: [B, 1, K, 2]
        
        # 3. Kích hoạt (Activation) để đảm bảo ý nghĩa vật lý
        # a (scale) > 0: Dùng softplus
        dyn_a = F.softplus(params[..., 0]) + 0.01 
        # b (translation) thuộc [0, seq_len]: Dùng sigmoid
        dyn_b = torch.sigmoid(params[..., 1]) * self.seq_len 
        
        # --- D. DUAL-INPUT KAN LEARNING ---
        
        # Nhánh 1: Base (Trend) - Học trên x_smooth
        base_out = self.wav_kan.forward_base(x_smooth_t)
        
        # Nhánh 2: Kan (Spike) - Học trên x_spike VỚI THAM SỐ ĐỘNG
        # LƯU Ý: Hàm forward_kan bên trong AdaptiveWaveletKANBlock phải nhận dyn_a, dyn_b
        kan_out = self.wav_kan.forward_kan(x_spike_t, dyn_a, dyn_b)
        
        # --- E. MULTIPLICATIVE FUSION (Hợp nhất Nhân) ---
        # Tính cổng điều biến biên độ từ Trend (base_out)
        # Nếu Trend lớn -> Gate lớn -> Spike được khuếch đại
        amplitude_scale = torch.sigmoid(self.amplitude_gate(base_out)) * 2.0
        
        # Công thức: Output = Trend + (Spike * Scale)
        x_out = base_out + (kan_out * amplitude_scale)
        
        x_out = self.norm(x_out + residual)
        
        return x_out