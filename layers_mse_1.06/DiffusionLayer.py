import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_2.jacobi_polynomials import rational_jacobi_polynomial

class JacobiRKAN(nn.Module):
    """
    Lớp kích hoạt Rational Jacobi KAN (Standard Version).
    """
    def __init__(self, degree=3):
        super(JacobiRKAN, self).__init__()
        # Giới hạn degree từ 1 đến 6 theo khuyến nghị
        self.degree = max(1, min(6, degree))
        
        # Các tham số học được
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.iota = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        # Normalize parameters
        normalized_alpha = F.elu(self.alpha) + 1.0 
        normalized_beta = F.elu(self.beta) + 1.0
        normalized_iota = F.softplus(self.iota)

        # Gọi hàm tính toán đa thức từ utils
        return rational_jacobi_polynomial(
            inputs, 
            self.degree, 
            normalized_alpha, 
            normalized_beta, 
            1, # zeta cố định là 1
            normalized_iota, 
            backend=torch
        )

class rKANBlock(nn.Module):
    """
    Block xây dựng cơ bản cho JD-KAN.
    Cấu trúc: Input -> [Linear Base] + [JacobiRKAN -> Linear Mixing] -> Norm
    """
    def __init__(self, input_dim, output_dim, kan_order=3):
        super(rKANBlock, self).__init__()
        
        # 1. Nhánh Linear truyền thống
        self.base_linear = nn.Linear(input_dim, output_dim)
        
        # 2. Nhánh Rational KAN
        self.rational_act = JacobiRKAN(degree=kan_order)
        self.kan_linear = nn.Linear(input_dim, output_dim)
        
        # Norm
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Dim]
        
        # Nhánh Base
        base = self.base_linear(x)
        base = F.silu(base)
        
        # Nhánh KAN
        kan = self.rational_act(x) 
        kan = self.kan_linear(kan)
        
        # Hợp nhất
        return self.norm(base + kan)

class DiffusionLayer(nn.Module):
    """
    Module 2: Continuous Dynamics Layer (JacobiRKAN-based)
    Purpose: Projects the smooth historical trend into the future.
    """
    def __init__(self, seq_len, pred_len, n_channels, d_model=64, rkan_order=3):
        super(DiffusionLayer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Encoding: Channel -> Feature Space
        self.encoder = nn.Linear(n_channels, d_model)
        
        # 2. Dynamics Modeling (Stacked rKAN Blocks)
        # Học quy luật biến đổi phi tuyến của Trend
        self.rkan_block1 = rKANBlock(d_model, d_model, kan_order=rkan_order)
        self.rkan_block2 = rKANBlock(d_model, d_model, kan_order=rkan_order)
        
        # 3. Time Projection (Extrapolation)
        # Chiếu từ quá khứ (Seq_Len) ra tương lai (Pred_Len)
        self.time_projector = nn.Linear(seq_len, pred_len)
        
        # 4. Decoding: Feature Space -> Channel
        self.decoder = nn.Linear(d_model, n_channels)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_smooth):
        """
        Input: x_smooth [Batch, Seq_Len, Channels] (Output from SmoothLayer)
        Output: future_cont [Batch, Pred_Len, Channels]
        """
        B, S, C = x_smooth.shape
        
        # --- Step 1: Encode ---
        # [B, S, C] -> [B, S, d_model]
        x_enc = self.encoder(x_smooth)
        x_enc = F.relu(x_enc)
        
        # --- Step 2: Learn Dynamics (Jacobi rKAN) ---
        # rKANBlock xử lý shape [B, S, D] trực tiếp (Last dim is feature)
        x_rkan = self.rkan_block1(x_enc)
        x_rkan = self.dropout(x_rkan)
        x_rkan = self.rkan_block2(x_rkan)
        x_rkan = self.dropout(x_rkan)
        
        # --- Step 3: Time Projection ---
        # Cần transpose để Linear layer tác động lên trục Time
        # [B, S, D] -> [B, D, S]
        x_rkan_T = x_rkan.permute(0, 2, 1)
        
        # Project: S -> P
        # [B, D, S] -> [B, D, P]
        x_future_T = self.time_projector(x_rkan_T)
        
        # Transpose back: [B, P, D]
        x_future_feat = x_future_T.permute(0, 2, 1)
        
        # --- Step 4: Decode ---
        # [B, P, D] -> [B, P, C]
        future_cont = self.decoder(x_future_feat)
        
        return future_cont