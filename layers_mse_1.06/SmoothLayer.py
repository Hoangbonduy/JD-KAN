import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MovingAvg(nn.Module):
    """
    Standard Moving Average Layer to extract naive trend.
    Used as the backbone/residual connection to ensure stability.
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, C, L]
        # Padding to keep same length
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x)
        return x

class AdaptiveSpectralBlock(nn.Module):
    """
    The Innovation: Learnable Frequency Block directly on raw values.
    Captures complex periodic patterns that Moving Average misses.
    """
    def __init__(self, seq_len, n_fourier_terms=10, d_model=64):
        super(AdaptiveSpectralBlock, self).__init__()
        self.n_terms = n_fourier_terms
        
        # Learnable Frequencies (Log-spaced initialization)
        init_freqs = torch.logspace(math.log10(0.01), math.log10(5.0), self.n_terms)
        self.omega = nn.Parameter(init_freqs.view(1, self.n_terms, 1)) 
        self.phase = nn.Parameter(torch.zeros(1, self.n_terms, 1))
        
        # Input dim: Time features (2*N) + Value features (2*N) = 4*N
        self.expanded_dim = 4 * self.n_terms
        
        self.gating_weight = nn.Parameter(torch.ones(1, self.expanded_dim, 1))
        
        # Convolutional Aggregator
        self.conv_block = nn.Sequential(
            nn.Conv1d(self.expanded_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        self.out_proj = nn.Conv1d(d_model, 1, kernel_size=1)
        
        # Zero Init to ensure it starts as a gentle correction, not a disruption
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_norm):
        # x_norm: [B*C, 1, S] - Normalized Input
        B_times_C, _, S = x_norm.shape
        
        # 1. Time Features (Adaptive Fourier on Time Index)
        t = torch.arange(S, device=x_norm.device, dtype=x_norm.dtype).view(1, 1, S)
        # Normalize t to range [0, 2pi] roughly for stability
        t_norm = t / S * 2 * math.pi
        
        angle_time = self.omega * t_norm + self.phase
        time_feat = torch.cat([torch.sin(angle_time), torch.cos(angle_time)], dim=1)
        time_feat = time_feat.expand(B_times_C, -1, -1)
        
        # 2. Value Features (KAN-style on Input Values)
        # Learn harmonics of the value itself
        multipliers = torch.arange(1, self.n_terms + 1, device=x_norm.device).view(1, self.n_terms, 1)
        angle_val = x_norm * multipliers
        val_feat = torch.cat([torch.sin(angle_val), torch.cos(angle_val)], dim=1)
        
        # 3. Combine & Gate
        combined_feat = torch.cat([time_feat, val_feat], dim=1)
        gate = torch.sigmoid(self.gating_weight)
        filtered_feat = combined_feat * gate
        
        # 4. Project
        out = self.out_proj(self.conv_block(filtered_feat))
        return out

class SmoothLayer(nn.Module):
    """
    Robust Adaptive Decomposition Layer.
    Structure: x_smooth = MovingAvg(x) + AdaptiveSpectralCorrection(x)
    """
    def __init__(self, n_channels, seq_len, n_fourier_terms=8, d_model=32, ma_kernel=25):
        super(SmoothLayer, self).__init__()
        self.n_channels = n_channels
        
        # Backbone 1: Classic Moving Average (Guarantees stability)
        # kernel_size can be odd, e.g., 25 for hourly data (approx 1 day)
        self.moving_avg = MovingAvg(kernel_size=ma_kernel, stride=1)
        
        # Backbone 2: Adaptive Spectral Correction (The "Paper Contribution")
        self.spectral_correction = AdaptiveSpectralBlock(
            seq_len=seq_len,
            n_fourier_terms=n_fourier_terms,
            d_model=d_model
        )

    def forward(self, x_raw):
        """
        Input: x_raw [B, S, C]
        Output: x_smooth, residual
        """
        B, S, C = x_raw.shape
        
        # 1. Base Trend via Moving Average
        # Permute for Conv1d: [B, C, S]
        x_in = x_raw.permute(0, 2, 1)
        x_base_trend = self.moving_avg(x_in) # [B, C, S]
        
        # 2. Learnable Correction (Spectral KAN)
        # Reshape for channel independence: [B*C, 1, S]
        x_reshape = x_in.reshape(B * C, 1, S)
        
        # Normalize (Instance Norm) to help Neural Net learn
        mean = x_reshape.mean(dim=2, keepdim=True)
        std = x_reshape.std(dim=2, keepdim=True) + 1e-5
        x_norm = (x_reshape - mean) / std
        
        # Calculate correction
        correction_norm = self.spectral_correction(x_norm) # [B*C, 1, S]
        
        # Denormalize correction
        correction = correction_norm * std # Scale back (offset is handled by adding to base)
        correction = correction.reshape(B, C, S)
        
        # 3. Combine: Smooth = Base Trend + Correction
        x_smooth = x_base_trend + correction
        
        # Permute back to [B, S, C]
        x_smooth = x_smooth.permute(0, 2, 1)
        
        # 4. Residual
        residual = x_raw - x_smooth
        
        return x_smooth, residual