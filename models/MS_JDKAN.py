import torch
import torch.nn as nn
import torch.fft
from layers.StandardNorm import Normalize
from layers.DilatedJumpLayer import DilatedJumpLayer


class FrequencyDecomp(nn.Module):
    """Tách tín hiệu thành low-freq và high-freq residuals qua FFT interpolation.
    Tương tự TimeKAN nhưng dùng cho JD-KAN."""
    
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            seq_len_low = self.seq_len // (self.down_sampling_window ** (self.down_sampling_layers - i))
            seq_len_high = self.seq_len // (self.down_sampling_window ** (self.down_sampling_layers - i - 1))
            out_high_res = self._freq_interp(
                out_low.transpose(1, 2), seq_len_low, seq_len_high
            ).transpose(1, 2)
            out_high_left = out_high - out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_high_left)
        out_level_list.reverse()
        return out_level_list

    def _freq_interp(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros(
            [x_fft.size(0), x_fft.size(1), target_len // 2 + 1],
            dtype=x_fft.dtype, device=x_fft.device
        )
        out_fft[:, :, :seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class JDKANMixingBlock(nn.Module):
    """Thay thế FrequencyMixing của TimeKAN: dùng DilatedJumpLayer ở mỗi scale."""
    
    def __init__(self, d_model, seq_len, down_sampling_window, down_sampling_layers,
                 kan_order, dropout, kernel_size=3):
        super().__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        
        # Block cho lowest resolution
        lowest_len = seq_len // (down_sampling_window ** down_sampling_layers)
        self.front_block = DilatedJumpLayer(
            d_model, kernel_size=kernel_size,
            dilations=[1, 2],  # Ít dilation hơn cho sequence ngắn
            kan_order=kan_order, dropout=dropout, iterations=1
        )
        
        # Blocks cho các resolution cao hơn
        self.front_blocks = nn.ModuleList([
            DilatedJumpLayer(
                d_model, kernel_size=kernel_size,
                dilations=[1, 2, 4],  # Nhiều dilation hơn cho sequence dài hơn
                kan_order=kan_order, dropout=dropout, iterations=1
            )
            for i in range(down_sampling_layers)
        ])

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high = self.front_blocks[i](out_high)
            seq_len_low = self.seq_len // (self.down_sampling_window ** (self.down_sampling_layers - i))
            seq_len_high = self.seq_len // (self.down_sampling_window ** (self.down_sampling_layers - i - 1))
            out_high_res = self._freq_interp(
                out_low.transpose(1, 2), seq_len_low, seq_len_high
            ).transpose(1, 2)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_low)
        out_level_list.reverse()
        return out_level_list

    def _freq_interp(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros(
            [x_fft.size(0), x_fft.size(1), target_len // 2 + 1],
            dtype=x_fft.dtype, device=x_fft.device
        )
        out_fft[:, :, :seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        
        # Multi-scale params (giống TimeKAN)
        self.down_sampling_window = getattr(configs, 'down_sampling_window', 2)
        self.down_sampling_layers = getattr(configs, 'down_sampling_layers', 2)
        self.channel_independence = getattr(configs, 'channel_independence', 1)
        
        # Normalize per resolution level (giống TimeKAN)
        self.normalize_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=False)
            for _ in range(self.down_sampling_layers + 1)
        ])
        
        # Channel-independent embedding: (1 -> d_model)
        self.enc_embedding = nn.Linear(1, configs.d_model)
        
        # Frequency Decomposition + JD-KAN Mixing blocks
        self.decomp_blocks = nn.ModuleList([
            FrequencyDecomp(configs.seq_len, self.down_sampling_window, self.down_sampling_layers)
            for _ in range(configs.e_layers)
        ])
        self.mixing_blocks = nn.ModuleList([
            JDKANMixingBlock(
                configs.d_model, configs.seq_len,
                self.down_sampling_window, self.down_sampling_layers,
                kan_order=getattr(configs, 'kan_order', 3),
                dropout=configs.dropout
            )
            for _ in range(configs.e_layers)
        ])
        
        # Output: project back
        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        self.predict_layer = nn.Linear(configs.seq_len, configs.pred_len)

    def _multi_scale_downsample(self, x_enc):
        """Tạo multi-resolution inputs qua AvgPool, giống TimeKAN."""
        down_pool = nn.AvgPool1d(self.down_sampling_window)
        x_enc = x_enc.permute(0, 2, 1)  # B,T,C -> B,C,T
        x_enc_ori = x_enc
        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]  # level 0: full resolution
        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        return x_enc_sampling_list

    def forecast(self, x_enc):
        # Step 1: Multi-scale downsampling
        x_enc_list = self._multi_scale_downsample(x_enc)
        
        # Step 2: Normalize per level + Channel independence reshape
        x_list = []
        B = x_enc_list[0].size(0)
        N = self.enc_in
        for i, x in enumerate(x_enc_list):
            # x: [B, T_i, N]
            x = self.normalize_layers[i](x, 'norm')
            # Channel independence: treat each channel separately
            T_i = x.size(1)
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T_i, 1)  # [B*N, T_i, 1]
            x_list.append(x)
        
        # Step 3: Embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x)  # [B*N, T_i, d_model]
            enc_out_list.append(enc_out)
        
        # Step 4: Stacked Decomp + Mixing (core of the model)
        for i in range(len(self.decomp_blocks)):
            enc_out_list = self.decomp_blocks[i](enc_out_list)
            enc_out_list = self.mixing_blocks[i](enc_out_list)
        
        # Step 5: Take highest resolution output, predict
        dec_out = enc_out_list[0]  # [B*N, seq_len, d_model]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)  # [B*N, pred_len, d_model]
        dec_out = self.projection_layer(dec_out)  # [B*N, pred_len, 1]
        dec_out = dec_out.reshape(B, N, self.pred_len).permute(0, 2, 1).contiguous()  # [B, pred_len, N]
        
        # Step 6: Denormalize
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out
        else:
            raise ValueError('Other tasks not implemented yet')