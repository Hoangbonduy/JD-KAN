import torch
import torch.nn as nn
import torch.fft
from layers.StandardNorm import Normalize
from layers.rKAN import rKANBlock

class FrequencyJumpLayer(nn.Module):
    """
    Xử lý tín hiệu trong miền tần số để bắt các đỉnh (Peaks).
    Nguyên lý: Đỉnh nhọn được tạo thành từ các tần số cao. 
    Chúng ta dùng rKAN để chỉnh sửa biên độ/pha của các tần số này.
    """
    def __init__(self, d_model, seq_len, kan_order=2, dropout=0.1):
        super(FrequencyJumpLayer, self).__init__()
        self.seq_len = seq_len
        
        # Chỉ xử lý 50% tần số đầu (vì FFT đối xứng), giúp giảm nhiễu cực tốt
        self.freq_len = seq_len // 2 + 1
        
        # KAN xử lý phần Thực (Real) và Ảo (Imag) riêng biệt
        # Điều này cho phép học cả Amplitude (độ cao đỉnh) và Phase (vị trí đỉnh)
        self.rkan_real = rKANBlock(d_model, d_model, kan_order=kan_order, dropout=dropout)
        self.rkan_imag = rKANBlock(d_model, d_model, kan_order=kan_order, dropout=dropout)
        
        # Một lớp Linear để trộn thông tin tần số
        self.freq_mixer = nn.Linear(self.freq_len, self.freq_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, L, C = x.shape
        
        # 1. Chuyển sang miền tần số (FFT)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho') # [B, Freq_Len, C]
        
        # 2. Tách phần Thực và Ảo
        real = x_fft.real
        imag = x_fft.imag
        
        # 3. Frequency Mixing (Trộn thông tin giữa các tần số)
        # Giúp mô hình hiểu mối quan hệ giữa các chu kỳ sóng
        real = self.freq_mixer(real.permute(0, 2, 1)).permute(0, 2, 1)
        imag = self.freq_mixer(imag.permute(0, 2, 1)).permute(0, 2, 1)

        # 4. Channel Mixing bằng rKAN (Học phi tuyến tính)
        real = self.rkan_real(real)
        imag = self.rkan_imag(imag)
        
        # 5. Tái tạo lại tín hiệu (Inverse FFT)
        x_fft_new = torch.complex(real, imag)
        x_out = torch.fft.irfft(x_fft_new, n=self.seq_len, dim=1, norm='ortho')
        
        # Residual connection với input gốc để giữ lại thông tin
        return x + x_out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Normalization
        self.normalize = Normalize(configs.enc_in, affine=True, non_norm=False)

        # Embedding: Project input features -> d_model
        self.enc_embedding = nn.Linear(configs.enc_in, configs.d_model)

        # ============================================================
        # A. GLOBAL TREND (Time Domain)
        # Giữ lại Linear vì nó quá tốt cho Trend dài hạn (như bản v2)
        # ============================================================
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        self.trend_linear.weight.data.fill_(1.0 / self.seq_len)
        self.trend_linear.bias.data.fill_(0.0)

        # ============================================================
        # B. LOCAL JUMP/PEAKS (Frequency Domain)
        # Đây là vũ khí mới để bắt đỉnh
        # ============================================================
        self.freq_layers = nn.ModuleList([
            FrequencyJumpLayer(configs.d_model, configs.seq_len, kan_order=2, dropout=configs.dropout)
            for _ in range(configs.e_layers) # Xếp chồng các lớp tần số
        ])
        
        # Output Projection cho nhánh Frequency
        self.freq_projector = nn.Linear(configs.d_model, configs.c_out)
        self.freq_out_linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 1. Normalize
        x_norm = self.normalize(x_enc, 'norm')

        # 2. Trend Prediction (Trực tiếp trên Time Domain)
        # Linear layer học xu hướng chính của chuỗi
        trend_out = self.trend_linear(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 3. Jump/Peak Prediction (Trên Frequency Domain)
        # Embedding
        x_freq = self.enc_embedding(x_norm)
        
        # Đi qua các lớp Frequency KAN
        for layer in self.freq_layers:
            x_freq = layer(x_freq)
            
        # Project về output
        jump_out = self.freq_projector(x_freq)
        jump_out = self.freq_out_linear(jump_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 4. Fusion: Trend + Jump
        # Trend nắm cái "gốc", Jump nắm cái "ngọn" (đỉnh)
        final_out = trend_out + jump_out
        
        # 5. Denormalize
        final_out = self.normalize(final_out, 'denorm')
        
        return final_out