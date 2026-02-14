import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_researching.StandardNorm import Normalize
from layers_researching.CausalFrequencyDecomp import CausalFrequencyDecomp
from layers_researching.AdaptiveWaveletKAN import AdaptiveWaveletKANLayer
from layers_researching.ChebyKANLayer import ChebyKANLayer

class PhaseAwareJDKANMixingBlock(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1, kernel_size=25, num_wavelets=4):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_wavelets = num_wavelets
        
        # 1. Phân rã (Giữ nguyên Replicate Padding)
        self.decomp = CausalFrequencyDecomp(kernel_size=kernel_size)
        
        # 2. Xử lý ngữ cảnh cục bộ (Lite Version)
        self.trend_local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, 
                                          groups=d_model, padding_mode='replicate')
        self.spike_local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, 
                                          groups=d_model, padding_mode='replicate')

        # 3. Hybrid M-KAN (Lite Version)
        self.trend_kan = ChebyKANLayer(input_dim=d_model, output_dim=d_model, degree=3)
        
        # --- [TỐI ƯU HÓA] LIGHTWEIGHT HYPERNET ---
        self.slope_detector = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, 
                                        groups=d_model, padding_mode='replicate')
        
        self.hyper_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_wavelets * 2) 
        )

        self.amplitude_gate = nn.Linear(d_model, d_model)

        self.adaptive_kan = AdaptiveWaveletKANLayer(d_model, d_model, seq_len, num_wavelets=num_wavelets)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        trend_part, spike_part = self.decomp(x) 
        
        # --- Lightweight Context Processing ---
        trend_in = trend_part.permute(0, 2, 1) 
        spike_in = spike_part.permute(0, 2, 1)
        
        trend_ctx = self.trend_local_conv(trend_in).permute(0, 2, 1)
        spike_ctx = self.spike_local_conv(spike_in).permute(0, 2, 1)
        
        # --- Trend Branch ---
        B, T, C = trend_ctx.shape
        trend_out = self.trend_kan(trend_ctx.reshape(-1, C)).reshape(B, T, C)
        
        # --- Spike Branch (Dynamic - Optimized) ---
        slope = self.slope_detector(trend_in) 
        
        slope_pool = slope.mean(dim=-1)       
        value_pool = trend_part.mean(dim=1)   
        
        hyper_in = torch.cat([slope_pool, value_pool], dim=-1) 
        params = self.hyper_proj(hyper_in)
        params = params.view(B, 1, self.num_wavelets, 2)
        
        dyn_a = F.softplus(params[..., 0]) + 0.01 
        dyn_b = torch.sigmoid(params[..., 1]) * self.seq_len 
        
        spike_out = self.adaptive_kan(spike_ctx, dyn_a, dyn_b)
        
        # --- Multiplicative Fusion ---
        amp_scale = torch.sigmoid(self.amplitude_gate(trend_out)) * 2.0
        out = trend_out + (spike_out * amp_scale)
        
        return self.norm(out + x)

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.down_sampling_window = getattr(configs, 'down_sampling_window', 2)
        self.down_sampling_layers = getattr(configs, 'down_sampling_layers', 2)
        
        self.cascade_decomp = CausalFrequencyDecomp(kernel_size=self.down_sampling_window * 2 + 1)
        self.down_pool = nn.AvgPool1d(kernel_size=self.down_sampling_window, stride=self.down_sampling_window)

        self.normalize_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=False)
            for _ in range(self.down_sampling_layers + 1)
        ])
        
        self.enc_embedding = nn.Linear(1, configs.d_model)
        
        self.scale_weights = nn.Parameter(torch.ones(self.down_sampling_layers + 1)) 

        self.model_scales = nn.ModuleList()
        # Vòng lặp này để lại giá trị i (ví dụ i=2)
        for i in range(self.down_sampling_layers + 1):
            current_seq_len = self.seq_len // (self.down_sampling_window ** i)
            scale_layers = nn.Sequential(*[
                PhaseAwareJDKANMixingBlock(
                    d_model=configs.d_model,
                    seq_len=current_seq_len,
                    dropout=configs.dropout,
                    kernel_size=self.down_sampling_window,
                    num_wavelets=4
                ) for _ in range(configs.e_layers)
            ])
            self.model_scales.append(scale_layers)
        
        self.projectors = nn.ModuleList([
            nn.Linear(configs.d_model, 1) for _ in range(self.down_sampling_layers + 1)
        ])
        
        # --- [SỬA LỖI TẠI ĐÂY] ---
        # Đổi biến chạy từ "_" thành "i" để nó cập nhật đúng giá trị cho từng layer
        self.predictors = nn.ModuleList([
            nn.Linear(self.seq_len // (self.down_sampling_window ** i), self.pred_len)
            for i in range(self.down_sampling_layers + 1) # <--- ĐÃ SỬA: dùng i, không dùng _
        ])

    def _multi_scale_downsample(self, x_enc):
        x_enc_list = [x_enc]
        current_x = x_enc
        for _ in range(self.down_sampling_layers):
            trend, _ = self.cascade_decomp(current_x)
            trend_t = trend.permute(0, 2, 1)
            next_x_t = self.down_pool(trend_t)
            next_x = next_x_t.permute(0, 2, 1)
            x_enc_list.append(next_x)
            current_x = next_x 
        return x_enc_list

    def forecast(self, x_enc):
        x_enc_list = self._multi_scale_downsample(x_enc)
        output_sum = 0
        norm_weights = F.softmax(self.scale_weights, dim=0)
        
        for i, x_scale in enumerate(x_enc_list):
            x_norm = self.normalize_layers[i](x_scale, 'norm')
            B, T_i, C = x_norm.shape
            x_reshaped = x_norm.permute(0, 2, 1).contiguous().reshape(B * C, T_i, 1)
            
            enc_out = self.enc_embedding(x_reshaped) 
            enc_out = self.model_scales[i](enc_out)
            
            dec_out = self.projectors[i](enc_out)
            # Tại đây i sẽ khớp với kích thước của predictor
            dec_out = self.predictors[i](dec_out.transpose(1, 2)).transpose(1, 2)
            dec_out = dec_out.reshape(B, C, self.pred_len).permute(0, 2, 1)
            
            dec_out = self.normalize_layers[i](dec_out, 'denorm')
            output_sum = output_sum + (dec_out * norm_weights[i])
            
        return output_sum 

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)