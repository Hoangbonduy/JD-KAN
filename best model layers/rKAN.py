import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.jacobi_polynomials import rational_jacobi_polynomial

class JacobiRKAN(nn.Module):
    """
    Lớp kích hoạt Rational Jacobi KAN.
    """
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
    """
    Block xây dựng cơ bản cho JD-KAN.
    """
    def __init__(self, input_dim, output_dim, kan_order=3, dropout=0.1):
        super(rKANBlock, self).__init__()
        
        self.base_linear = nn.Linear(input_dim, output_dim)
        self.rational_act = JacobiRKAN(degree=kan_order)
        self.kan_linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Dim]
        
        # Nhánh Base
        base = self.base_linear(x)
        base = F.silu(base)
        
        # Nhánh KAN
        kan = self.rational_act(x) 
        kan = self.kan_linear(kan)
        
        # Hợp nhất và chuẩn hóa
        # Residual Connection có sẵn trong logic: base + kan
        out = self.norm(base + kan)
        return self.dropout(out)

class StackedrKAN(nn.Module):
    """
    Lớp xếp chồng nhiều rKANBlock cho Deep Continuous Stream.
    Có thêm Residual Connection giữa các block (x = Block(x) + x).
    """
    def __init__(self, d_model, layers=2, degree=3, dropout=0.1):
        super(StackedrKAN, self).__init__()
        self.blocks = nn.ModuleList([
            rKANBlock(d_model, d_model, degree, dropout) for _ in range(layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            # Residual connection giữa các lớp rKANBlock
            x = block(x) + x 
        return x