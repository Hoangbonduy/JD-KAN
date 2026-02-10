import torch
import torch.nn as nn
import torch.fft
from layers.StandardNorm import Normalize
from layers.DilatedJumpLayer import DilatedJumpLayer

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Norm giữ nguyên
        self.normalize = Normalize(configs.enc_in, affine=True, non_norm=False)
        
        # Embedding
        self.enc_embedding = nn.Linear(configs.enc_in, configs.d_model)
        
        # Không cần trend_linear nữa! (Loại bỏ để buộc học jumps)
        
        # Dilated layers (stack e_layers)
        self.dilated_layers = nn.ModuleList([
            DilatedJumpLayer(configs.d_model, kernel_size=3, dilations=[1,2,4,8,16], 
                             kan_order=configs.kan_order, dropout=configs.dropout)
            for _ in range(configs.e_layers)  # e_layers=3 từ script
        ])
        
        # Output projection
        self.projector = nn.Linear(configs.d_model, configs.c_out)
        self.out_linear = nn.Linear(self.seq_len, self.pred_len)  # Để match pred_len

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Norm
        x_norm = self.normalize(x_enc, 'norm')
        
        # Embedding
        x = self.enc_embedding(x_norm)
        
        # Đi qua dilated layers (learn multi-scale jumps + trend)
        for layer in self.dilated_layers:
            x = layer(x)
        
        # Project output
        out = self.projector(x)
        out = self.out_linear(out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Denorm
        final_out = self.normalize(out, 'denorm')
        
        return final_out