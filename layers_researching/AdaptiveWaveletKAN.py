import torch
import torch.nn as nn
import math

class AdaptiveWaveletKANLayer(nn.Module):
    def __init__(self, in_features, out_features, seq_len, num_wavelets=3):
        super().__init__()
        self.seq_len = seq_len
        self.num_wavelets = num_wavelets
        
        self.linear = nn.Linear(in_features, out_features)
        
        self.w = nn.Parameter(torch.empty(out_features, num_wavelets))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def get_wavelet_basis(self, x, a_dynamic, b_dynamic):
        batch_size = x.shape[0]
        device = x.device
        
        # t vector chuẩn
        t = torch.arange(self.seq_len, device=device).float().view(1, self.seq_len, 1)
        
        a = a_dynamic.unsqueeze(1)
        b = b_dynamic.unsqueeze(1)
        w = self.w.view(1, 1, -1, self.num_wavelets)
        
        t_expanded = t.unsqueeze(-1) 
        
        # Wavelet formula
        x_scaled = (t_expanded - b) / (a + 1e-5)
        
        # Mexican Hat
        psi = (1 - x_scaled**2) * torch.exp(-0.5 * x_scaled**2)
        
        phi_t = (w * psi).sum(dim=-1) 
        
        return phi_t

    def forward(self, x, dynamic_a, dynamic_b):
        out_linear = self.linear(x)
        phi_t = self.get_wavelet_basis(x, dynamic_a, dynamic_b)
        
        # Giữ lại phép cộng (Residual) này giúp gradient chảy tốt hơn
        return out_linear * phi_t + phi_t