import torch
import torch.nn as nn
import torch.nn.functional as F
from .jacobi_polynomials import rational_jacobi_polynomial

class JacobiRKAN(nn.Module):
    """
    Lớp kích hoạt Rational Jacobi KAN.
    Input: Tensor bất kỳ
    Output: Tensor cùng kích thước đã qua biến đổi phi tuyến
    """
    def __init__(self, degree=3):
        super(JacobiRKAN, self).__init__()
        # Giới hạn degree từ 1 đến 6 theo khuyến nghị của tác giả
        self.degree = max(1, min(6, degree))
        
        # Các tham số học được của hàm Rational
        # alpha, beta, iota khởi tạo là 1.0
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.iota = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        # Normalize parameters để đảm bảo miền giá trị ổn định
        # Sử dụng ELU + 1 để đảm bảo dương
        normalized_alpha = F.elu(self.alpha) + 1.0 
        normalized_beta = F.elu(self.beta) + 1.0
        normalized_iota = F.softplus(self.iota) # softplus luôn dương

        # Gọi hàm tính toán đa thức từ file utils
        return rational_jacobi_polynomial(
            inputs, 
            self.degree, 
            normalized_alpha, 
            normalized_beta, 
            1, # zeta cố định là 1 trong cài đặt chuẩn
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
        
        # 1. Nhánh Linear truyền thống (giống skip connection có trọng số)
        # Giúp gradient chảy tốt trong giai đoạn đầu training
        self.base_linear = nn.Linear(input_dim, output_dim)
        
        # 2. Nhánh Rational KAN (Học phi tuyến phức tạp)
        # Activation function là JacobiRKAN
        self.rational_act = JacobiRKAN(degree=kan_order)
        
        # Sau khi qua hàm kích hoạt, cần một lớp Linear để trộn thông tin các kênh
        # Đây là sự khác biệt chính so với KAN gốc (lưới tham số), giúp giảm số lượng tham số
        self.kan_linear = nn.Linear(input_dim, output_dim)
        
        # Normalization và Activation cuối
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Dim]
        
        # Nhánh Base
        base = self.base_linear(x)
        base = F.silu(base) # SiLU thường tốt cho Time Series
        
        # Nhánh KAN
        # Biến đổi phi tuyến từng phần tử
        kan = self.rational_act(x) 
        # Trộn kênh
        kan = self.kan_linear(kan)
        
        # Hợp nhất và chuẩn hóa
        # Cộng gộp giúp model học được cả tính chất tuyến tính và phi tuyến
        return self.norm(base + kan)