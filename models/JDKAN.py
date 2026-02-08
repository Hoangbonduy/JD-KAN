import torch
import torch.nn as nn
from layers.StandardNorm import Normalize
from layers.DecompLayer import CascadedDecomp
from layers.DiffusionLayer import DiffusionLayer
from layers.JumpLayer import JumpLayer
from layers.FusionLayer import FusionLayer

class Model(nn.Module):
    """
    JD-KAN: Jump-Diffusion Kolmogorov-Arnold Network
    
    Mô hình dự báo chuỗi thời gian thế hệ mới, kết hợp:
    1. Continuous Flow: Sử dụng rKAN (Rational KAN) để ngoại suy xu hướng mượt mà.
    2. Discrete Jumps: Sử dụng Memory Bank để phát hiện và tái tạo các cú sốc bất ngờ.
    3. Adaptive Fusion: Cơ chế trộn thông minh giữa hai luồng tín hiệu.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # 1. Normalization (RevIN)
        # Giúp model học tốt hơn trên dữ liệu non-stationary (biến động phân phối)
        self.normalize = Normalize(configs.enc_in, affine=True, non_norm=False)
        
        # 2. Decomposition
        # Tách dữ liệu thành các thành phần tần số khác nhau
        self.decomposition = CascadedDecomp(kernel_low=25, kernel_mid=11)
        
        # 3. Branch 1: Continuous Stream (Trend + Season)
        # Dùng rKAN để học động lực học liên tục
        # kan_order: Bậc của đa thức (được config truyền vào)
        self.continuous_stream = DiffusionLayer(
            input_dim=configs.enc_in,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            kan_order=configs.kan_order if hasattr(configs, 'kan_order') else 3
        )
        
        # 4. Branch 2: Jump Stream (Residuals -> Events)
        # Dùng Memory Bank để "nhớ" các mẫu hình shock
        # n_jumps: Số lượng mẫu trong bộ nhớ (mặc định 32 nếu không config)
        self.jump_stream = JumpLayer(
            input_dim=configs.enc_in,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            n_jumps=32,   # Có thể đưa vào configs nếu cần tuning
            d_model=64    # Hidden dim cho Gating Network
        )
        
        # 5. Fusion Layer
        # Hợp nhất hai nhánh
        self.fusion = FusionLayer(d_model=configs.enc_in, dropout=configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [Batch, Seq_Len, Channels]
        
        # --- Bước 1: Chuẩn hóa (Normalization) ---
        # "seq_last" nghĩa là chuẩn hóa theo chiều thời gian (phổ biến cho Time Series)
        x_norm = self.normalize(x_enc, 'norm')

        # --- Bước 2: Phân rã (Decomposition) ---
        # Tách x_norm thành 3 phần
        x_trend, x_season, x_resid = self.decomposition(x_norm)

        # --- Bước 3: Xử lý các nhánh (Processing Branches) ---
        
        # Nhánh 1: Dự báo phần liên tục (Continuous)
        # Output: [Batch, Pred_Len, Channels]
        future_cont = self.continuous_stream(x_trend, x_season)
        
        # Nhánh 2: Dự báo phần nhảy vọt (Jumps)
        # Output: [Batch, Pred_Len, Channels]
        future_jump = self.jump_stream(x_resid)

        # --- Bước 4: Hợp nhất (Fusion) ---
        # Output: [Batch, Pred_Len, Channels]
        future_final = self.fusion(future_cont, future_jump)

        # --- Bước 5: Trả lại scale gốc (Denormalization) ---
        y_pred = self.normalize(future_final, 'denorm')
        
        return y_pred