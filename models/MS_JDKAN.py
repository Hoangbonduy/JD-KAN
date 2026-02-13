import torch
import torch.nn as nn
from layers.StandardNorm import Normalize
from layers.CausalFrequencyDecomp import CausalFrequencyDecomp
from layers.AdaptiveWaveletKAN import AdaptiveWaveletKANBlock

class JDKANMixingBlock(nn.Module):
    """
    Block trộn thông tin: Tách Trend/Spike -> Xử lý riêng -> Gated Fusion
    """
    def __init__(self, d_model, seq_len, dropout=0.1, kernel_size=25, num_wavelets=4):
        super().__init__()
        # 1. Bộ phân rã mới (Causal) - Thay thế FFT
        self.decomp = CausalFrequencyDecomp(kernel_size=kernel_size)
        
        # 2. Nhánh Trend (Linear đơn giản để bắt xu hướng mượt)
        self.trend_conv = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Nhánh Spike (Adaptive Wavelet KAN - "Ngôi sao" của mô hình)
        # Tự động học vị trí (b_k) và độ rộng (a_k) của các gai
        self.adaptive_kan = AdaptiveWaveletKANBlock(
            input_dim=d_model, 
            output_dim=d_model, 
            seq_len=seq_len, 
            dropout=dropout, 
            num_wavelets=num_wavelets
        )
        
        # 4. Gated Fusion (Cổng hợp nhất)
        self.gate_linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [Batch, Seq, Channel]
        
        # Bước 1: Tách Trend và Spike (Causal)
        # Lưu ý: decomp trả về [B, Seq, C] do đã permute bên trong class CausalFrequencyDecomp
        trend_part, spike_part = self.decomp(x)
        
        # Bước 2: Xử lý song song
        trend_out = self.trend_conv(trend_part)
        spike_out = self.adaptive_kan(spike_part) 
        
        # Bước 3: Gated Fusion 
        # Trend quyết định độ mở của cổng cho Spike
        # (Nếu Trend cao -> Khả năng Spike lớn là cao -> Mở cổng)
        gate = torch.sigmoid(self.gate_linear(trend_out))
        
        # Hợp nhất: Trend + (Spike * Gate)
        out = trend_out + (spike_out * gate)
        
        # Residual connection với input ban đầu
        return self.norm(out + x)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Cấu hình Multi-scale
        self.down_sampling_window = getattr(configs, 'down_sampling_window', 2)
        self.down_sampling_layers = getattr(configs, 'down_sampling_layers', 2)
        
        # 1. Normalization (RevIN) riêng cho từng Level phân giải
        # Level 0: Gốc, Level 1: Mịn hơn, Level 2: Xu hướng
        self.normalize_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=False)
            for _ in range(self.down_sampling_layers + 1)
        ])
        
        # 2. Embedding: Input 1 chiều -> D_model
        # Dùng chung một lớp Linear này cho tất cả các scale (Weight Sharing) để tiết kiệm bộ nhớ
        self.enc_embedding = nn.Linear(1, configs.d_model)
        
        # 3. Encoder Blocks (AW-KAN) - CỐT LÕI CỦA MULTI-SCALE
        # Ta tạo các block xử lý riêng biệt cho từng scale.
        # Lý do: Gai nhọn (scale 0) cần bộ lọc Wavelet khác hẳn với Trend (scale 2).
        self.model_scales = nn.ModuleList()
        for i in range(self.down_sampling_layers + 1):
            # Tính độ dài chuỗi tại scale này (ví dụ: 96 -> 48 -> 24)
            current_seq_len = self.seq_len // (self.down_sampling_window ** i)
            
            # Tạo chuỗi các lớp AW-KAN cho scale i
            scale_layers = nn.Sequential(*[
                JDKANMixingBlock(
                    d_model=configs.d_model,
                    seq_len=current_seq_len, # Quan trọng: Truyền đúng độ dài chuỗi đã giảm
                    dropout=configs.dropout,
                    kernel_size=self.down_sampling_window,
                    num_wavelets=4
                ) for _ in range(configs.e_layers)
            ])
            self.model_scales.append(scale_layers)
        
        # 4. Projectors & Predictors (Riêng cho từng scale)
        self.projectors = nn.ModuleList([
            nn.Linear(configs.d_model, 1)
            for _ in range(self.down_sampling_layers + 1)
        ])
        
        self.predictors = nn.ModuleList([
            nn.Linear(
                self.seq_len // (self.down_sampling_window ** i), 
                self.pred_len
            )
            for i in range(self.down_sampling_layers + 1)
        ])

    def _multi_scale_downsample(self, x_enc):
        """Tạo list các input với độ phân giải giảm dần bằng AvgPool"""
        # x_enc: [B, T, C] -> Permute [B, C, T] để Pooling
        x_perm = x_enc.permute(0, 2, 1)
        
        x_list = [x_perm] # Level 0 (Gốc)
        current_x = x_perm
        
        pool = nn.AvgPool1d(self.down_sampling_window)
        for _ in range(self.down_sampling_layers):
            current_x = pool(current_x)
            x_list.append(current_x)
            
        # Trả về dạng [B, T, C] cho thống nhất
        return [x.permute(0, 2, 1) for x in x_list]

    def forecast(self, x_enc):
        # --- BƯỚC 1: Tạo Multi-scale Inputs ---
        # x_enc_list chứa [Scale0(Gốc), Scale1(Mịn), Scale2(Rất mịn)]
        x_enc_list = self._multi_scale_downsample(x_enc)
        
        output_sum = 0
        
        # --- BƯỚC 2: Vòng lặp xử lý từng Scale ---
        for i, x_scale in enumerate(x_enc_list):
            # A. Normalize (RevIN) riêng từng scale
            # x_scale: [B, Seq_i, C]
            x_norm = self.normalize_layers[i](x_scale, 'norm')
            
            # --- CHANNEL INDEPENDENCE (Quan trọng) ---
            B, T_i, C = x_norm.shape
            # Biến đổi: [B, T_i, C] -> [B*C, T_i, 1]
            # Mẹo: Gộp Batch và Channel lại để model coi mỗi channel là một sample độc lập
            x_reshaped = x_norm.permute(0, 2, 1).contiguous().reshape(B * C, T_i, 1)
            
            # B. Embedding & Processing (AW-KAN)
            enc_out = self.enc_embedding(x_reshaped) # [B*C, T_i, D]
            
            # Chạy qua các block AW-KAN tương ứng với scale này
            enc_out = self.model_scales[i](enc_out)
            
            # C. Prediction
            # Project về 1 chiều: [B*C, T_i, 1]
            dec_out = self.projectors[i](enc_out)
            
            # Phóng ra tương lai (Linear): [B*C, T_i, 1] -> [B*C, Pred, 1]
            dec_out = self.predictors[i](dec_out.transpose(1, 2)).transpose(1, 2)
            
            # Reshape lại để tách Batch và Channel: [B*C, Pred, 1] -> [B, Pred, C]
            dec_out = dec_out.reshape(B, C, self.pred_len).permute(0, 2, 1)
            
            # D. Denormalize
            dec_out = self.normalize_layers[i](dec_out, 'denorm')
            
            # E. Cộng dồn kết quả (Ensemble)
            output_sum = output_sum + dec_out
            
        # Chia trung bình kết quả từ các scale
        return output_sum / len(x_enc_list)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)