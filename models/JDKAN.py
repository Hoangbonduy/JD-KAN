import torch
import torch.nn as nn
from layers.StandardNorm import Normalize
from layers.SmoothLayer import SmoothLayer
from layers.DiffusionLayer import DiffusionLayer
from layers.JumpLayer import JumpLayer
from layers.FusionLayer import FusionLayer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Adaptive Spectral-KAN Decomposer
        # Phân rã thành x_smooth (Trend + Seasonality) và residual (Noise + Jumps)
        self.decomposition = SmoothLayer(
            n_channels=configs.enc_in,
            seq_len=configs.seq_len,
            n_fourier_terms=getattr(configs, 'n_fourier_terms', 8),
            d_model=getattr(configs, 'd_model', 256) // 8,  # Use smaller d_model for SmoothLayer
            ma_kernel=25
        )

        self.continuous_stream = DiffusionLayer(
            input_dim=configs.enc_in,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            kan_order=configs.kan_order
        )

        self.jump_stream = JumpLayer(
            input_dim=configs.enc_in,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            d_model=configs.d_model
        )

        self.fusion = FusionLayer(configs.enc_in)
        self.normalize = Normalize(configs.enc_in, affine=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_norm = self.normalize(x_enc, 'norm')

        # Phân rã: Smooth (Trend + Seasonality) và Residual (Noise + Jumps)
        x_smooth, residual = self.decomposition(x_norm)

        # Nhánh Continuous học Smooth component
        future_cont = self.continuous_stream(x_smooth)
        
        # Nhánh Jump học Residual component (high-frequency)
        future_jump = self.jump_stream(residual)

        # Debug Fusion Layer
        if self.training == False: # Chỉ vẽ khi test/valid để không làm chậm train
             import matplotlib.pyplot as plt
             import os
             if not os.path.exists("./debug_fusion"): os.makedirs("./debug_fusion")
             
             # Lấy sample đầu tiên, channel cuối cùng (thường khó dự báo nhất)
             idx = 0
             chn = -1 
             
             pred_cont = future_cont[idx, :, chn].detach().cpu().numpy()
             pred_jump = future_jump[idx, :, chn].detach().cpu().numpy()
             combined = (future_cont + future_jump)[idx, :, chn].detach().cpu().numpy()
             
             plt.figure(figsize=(10, 6))
             plt.plot(pred_cont, label='Continuous (Trend+Season)', linewidth=2)
             plt.plot(pred_jump, label='Jump Stream', color='red', alpha=0.7)
             plt.plot(combined, label='Fused Result', linestyle='--')
             plt.title("Fusion Layer Inspection: Components Breakdown")
             plt.legend()
             plt.grid(True)
             plt.savefig(f"./debug_fusion/fusion_step.png")
             plt.close()

        future_final = self.fusion(future_cont, future_jump)
        y_pred = self.normalize(future_final, 'denorm')

        return y_pred