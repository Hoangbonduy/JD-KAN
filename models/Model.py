import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from layers_2.SmoothLayer import SmoothLayer
from layers_2.DiffusionLayer import DiffusionLayer
from layers_2.JumpLayer import JumpLayer
from layers_2.FusionLayer import FusionLayer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        
        # --- KHẮC PHỤC 1: GIẢM ĐỘ PHỨC TẠP ---
        # ETTh1 là dataset nhỏ. d_model=256 là quá thừa thãi -> gây Overfitting.
        # Ép xuống 64 để model buộc phải học quy luật thay vì học vẹt.
        target_d_model = 64 
        
        # --- 1. Decomposition Layer ---
        self.decomposition = SmoothLayer(
            n_channels=configs.enc_in,
            seq_len=configs.seq_len,
            n_fourier_terms=8, 
            d_model=target_d_model
        )

        # --- 2. Continuous Stream ---
        self.continuous_stream = DiffusionLayer(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            n_channels=configs.enc_in,
            d_model=target_d_model,
            rkan_order=3 # Giữ order thấp để ổn định
        )

        # --- 3. Jump Stream ---
        self.jump_stream = JumpLayer(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            n_channels=configs.enc_in,
            d_model=target_d_model,
            dropout=configs.dropout
        )

        # --- 4. Fusion Layer ---
        self.fusion = FusionLayer(
            n_channels=configs.enc_in,
            d_model=target_d_model
        )
        
        # --- KHẮC PHỤC 2: BỎ LỚP NORMALIZE (RevIN) BAO NGOÀI ---
        # self.normalize = Normalize(configs.enc_in, affine=True) <--- XÓA HOẶC COMMENT DÒNG NÀY

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # --- KHẮC PHỤC 2: TRUYỀN THẲNG DỮ LIỆU GỐC ---
        # x_norm = self.normalize(x_enc, 'norm') <--- KHÔNG DÙNG CÁI NÀY NỮA
        
        # Bước 1: Phân rã (Decomposition)
        # SmoothLayer cần nhìn thấy Trend thật (dốc lên/xuống) để Moving Average hoạt động đúng.
        x_smooth, x_resid = self.decomposition(x_enc) 

        # Bước 2: Dự báo
        future_cont = self.continuous_stream(x_smooth)
        future_jump = self.jump_stream(x_resid)

        # # Bước 3: Debug (Chỉ chạy khi Test)
        # if self.training == False: 
        #     if not os.path.exists("./debug_fusion"): os.makedirs("./debug_fusion")
        #     idx = 0
        #     chn = -1 
        #     pred_cont = future_cont[idx, :, chn].detach().cpu().numpy()
        #     pred_jump = future_jump[idx, :, chn].detach().cpu().numpy()
            
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(pred_cont, label='Continuous', linewidth=2)
        #     plt.plot(pred_jump, label='Jump', color='red', alpha=0.7)
        #     plt.title("Fixed Model - Components Breakdown")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(f"./debug_fusion/fusion_check.png")
        #     plt.close()

        # Bước 4: Fusion
        future_final = self.fusion(future_cont, future_jump)
        
        # Trả về trực tiếp (Không Denorm)
        return future_final