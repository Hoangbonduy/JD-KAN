import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.jacobi_polynomials import rational_jacobi_polynomial

class JacobiRKAN(nn.Module):
    """Rational Jacobi KAN activation layer."""
    def __init__(self, degree=3):
        super(JacobiRKAN, self).__init__()
        self.degree = max(1, min(6, degree))
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.iota = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        normalized_alpha = F.elu(self.alpha) + 1.0 
        normalized_beta = F.elu(self.beta) + 1.0
        normalized_iota = F.softplus(self.iota)
        
        return rational_jacobi_polynomial(
            inputs, 
            self.degree, 
            normalized_alpha, 
            normalized_beta, 
            1, 
            normalized_iota, 
            backend=torch
        )

class rKANBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kan_order=3, dropout=0.1, num_heads=4):  # Thêm num_heads
        super(rKANBlock, self).__init__()
        
        self.base_linear = nn.Linear(input_dim, output_dim)
        self.base_activation = nn.SiLU()
        
        self.rational_act = JacobiRKAN(degree=kan_order)
        self.kan_linear = nn.Linear(input_dim, output_dim)
        
        # THÊM: Cross-Attention giữa base và kan outputs
        self.cross_attn = nn.MultiheadAttention(output_dim, num_heads=num_heads, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base = self.base_activation(self.base_linear(x))
        kan = self.rational_act(self.kan_linear(x))
        
        # Cross-attention: Query=kan (spike), Key/Value=base (smooth)
        attended, _ = self.cross_attn(kan.transpose(0,1), base.transpose(0,1), base.transpose(0,1))
        attended = attended.transpose(0,1)
        
        # Tổng hợp: attended + base (hoặc + kan tùy tune)
        return self.dropout(attended + base)
    
    def forward_base(self, x):
        """Chỉ chạy nhánh Base (Linear)"""
        return self.base_activation(self.base_linear(x))
    
    def forward_kan(self, x):
        """Chỉ chạy nhánh KAN (Jacobi)"""
        return self.rational_act(self.kan_linear(x))