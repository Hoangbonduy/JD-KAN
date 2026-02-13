import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveWaveletKANLayer(nn.Module):
    """
    Lớp Adaptive Wavelet KAN dựa trên thời gian (Time-Domain).
    Thay vì biến đổi giá trị input, lớp này học các trọng số biến thiên theo thời gian
    (Time-Varying Weights) để điều biến tín hiệu.
    
    Công thức: Output(t) = Linear(Input(t)) * Sum(w_k * psi((t - b_k) / a_k))
    """
    def __init__(self, in_features, out_features, seq_len, num_wavelets=3, wavelet_type='mexican_hat'):
        super().__init__()
        self.seq_len = seq_len
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type
        
        # 1. Linear Transformation để trộn kênh input (Channel Mixing)
        self.linear = nn.Linear(in_features, out_features)
        
        # 2. Các tham số Adaptive Wavelet (Learnable Time-Frequency Basis)
        # Shape: [Out_features, Num_wavelets] - Mỗi kênh đầu ra có bộ wavelet riêng
        self.a = nn.Parameter(torch.empty(out_features, num_wavelets)) # Scale (Độ rộng)
        self.b = nn.Parameter(torch.empty(out_features, num_wavelets)) # Translation (Vị trí)
        self.w = nn.Parameter(torch.empty(out_features, num_wavelets)) # Weight (Trọng số đóng góp)
        
        self._init_parameters()

    def _init_parameters(self):
        # --- Khởi tạo theo hướng dẫn trong ảnh ---
        
        # 1. Khởi tạo a_k (Scale): Thang Logarit từ 1.0 đến T/4
        # log(1) = 0, log(T/4) = log_max
        T = self.seq_len
        log_min = 0.0
        log_max = math.log(max(T / 4, 1.0)) # Đảm bảo không âm
        
        # Tạo grid logarit
        a_init = torch.logspace(log_min, log_max, self.num_wavelets, base=math.e)
        # Mở rộng cho mọi out_features
        self.a.data = a_init.unsqueeze(0).expand(self.a.shape).clone()
        
        # 2. Khởi tạo b_k (Translation): Rải đều tuyến tính từ 0 đến T-1
        b_init = torch.linspace(0, T - 1, self.num_wavelets)
        self.b.data = b_init.unsqueeze(0).expand(self.b.shape).clone()
        
        # 3. Khởi tạo w_k (Weight): Kaiming Uniform hoặc Normal
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        
        # Init Linear layer
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def get_wavelet_basis(self, device):
        """Tính toán giá trị phi(t) cho toàn bộ chuỗi thời gian"""
        # Tạo vector thời gian t: [1, Seq_len, 1]
        t = torch.arange(self.seq_len, device=device).float().view(1, self.seq_len, 1)
        
        # Reshape tham số để broadcasting: [Out, Wavelets] -> [1, 1, Out, Wavelets]
        a = self.a.view(1, 1, -1, self.num_wavelets)
        b = self.b.view(1, 1, -1, self.num_wavelets)
        w = self.w.view(1, 1, -1, self.num_wavelets)
        
        # Tính toán (t - b) / a
        # t mở rộng thành [1, Seq, 1, 1]
        t_expanded = t.unsqueeze(-1) 
        x_scaled = (t_expanded - b) / (a + 1e-5) # Cộng epsilon tránh chia 0
        
        # Tính psi(x)
        if self.wavelet_type == 'mexican_hat':
            term1 = (x_scaled ** 2) - 1
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            psi = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            psi = torch.cos(5.0 * x_scaled) * torch.exp(-0.5 * x_scaled ** 2)
        else: # Fallback Gaussian
             psi = torch.exp(-0.5 * x_scaled ** 2)
             
        # Tổng hợp các wavelet: Sum(w_k * psi_k)
        # Kết quả phi_t: [1, Seq, Out]
        phi_t = (w * psi).sum(dim=-1)
        
        return phi_t

    def forward(self, x):
        # x shape: [Batch, Seq_len, In_features]
        
        # Bước 1: Trộn thông tin các kênh (Linear Mixing)
        # out_linear: [Batch, Seq_len, Out_features]
        out_linear = self.linear(x)
        
        # Bước 2: Tạo Time-Domain Basis
        # phi_t: [1, Seq_len, Out_features]
        phi_t = self.get_wavelet_basis(x.device)
        
        # Bước 3: Modulation (Nhân element-wise)
        # Tín hiệu được khuếch đại hoặc triệt tiêu dựa trên vị trí thời gian
        out = out_linear * phi_t
        
        return out

class AdaptiveWaveletKANBlock(nn.Module):
    """
    Block tích hợp thay thế cho WaveletKANBlock cũ.
    Kết hợp nhánh Base (Trend) và nhánh Adaptive Wavelet (Spike/Detail).
    """
    def __init__(self, input_dim, output_dim, seq_len, dropout=0.1, num_wavelets=3):
        super().__init__()
        
        # Nhánh Base: Linear + SiLU (Học các biến đổi mượt/phi tuyến thông thường)
        self.base_linear = nn.Linear(input_dim, output_dim)
        self.base_activation = nn.SiLU()
        
        # Nhánh Adaptive KAN: Học các đặc trưng phụ thuộc thời gian (Time-Aware)
        # Đây là phần thay thế quan trọng theo ý tưởng của bạn
        self.adaptive_kan = AdaptiveWaveletKANLayer(
            input_dim, output_dim, seq_len, num_wavelets=num_wavelets
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: [Batch, Seq, Channel] - Lưu ý: Input phải ở dạng [B, L, C]
        
        # 1. Nhánh Base (Smooth/Global features)
        base = self.base_linear(x)
        base = self.base_activation(base)
        
        # 2. Nhánh Adaptive Wavelet (Time-localized features)
        kan = self.adaptive_kan(x)
        
        # 3. Gated Fusion (Cải tiến so với cộng đơn thuần)
        # Cho phép mô hình tự học cách cân bằng giữa Trend và Spike tại mỗi thời điểm
        out = base + kan 
        
        return self.layer_norm(self.dropout(out))