import torch
import torch.nn as nn

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        
        # --- [SỬA LỖI QUAN TRỌNG] ---
        # Kẹp giá trị để tránh lỗi NaN tại hàm acos khi xấp xỉ 1.0 hoặc -1.0
        x = torch.clamp(x, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        # ----------------------------

        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        
        # Apply acos
        x = x.acos()
        
        # Multiply by arange [0 .. degree]
        x *= self.arange
        
        # Apply cos
        x = x.cos()
        
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        
        y = y.view(-1, self.outdim)
        return y