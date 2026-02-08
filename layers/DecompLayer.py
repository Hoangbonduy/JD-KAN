import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    """
    Moving Average block to highlight the trend of time series.
    Logic lấy từ Autoformer/TimeKAN: Dùng AvgPool1d với padding khéo léo 
    để giữ nguyên độ dài chuỗi (Seq_Len).
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]
        
        # Padding vào 2 đầu chuỗi để khi Pooling không bị giảm kích thước
        # front: lặp lại phần tử đầu tiên
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # end: lặp lại phần tử cuối cùng
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        
        # Nối lại: [front, x, end]
        x = torch.cat([front, x, end], dim=1)
        
        # AvgPool chạy trên chiều thứ 2 (nên phải permute)
        x = x.permute(0, 2, 1) 
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        
        return x

class SeriesDecomp(nn.Module):
    """
    Series decomposition block cơ bản.
    Tách chuỗi thành: Trend (Moving Avg) + Residual (Phần còn lại)
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class CascadedDecomp(nn.Module):
    """
    Phân rã đa tầng (Cascaded Frequency Decomposition - CFD) cho JD-KAN.
    
    Quy trình:
    1. Input -> [Low-pass Filter (Large Kernel)] -> Trend (Low Freq)
       Phần dư 1 = Input - Trend
       
    2. Phần dư 1 -> [Mid-pass Filter (Small Kernel)] -> Seasonality (Mid Freq)
       Phần dư 2 (Residual) = Phần dư 1 - Seasonality
       
    Output cuối cùng:
    - x_trend: Đi vào rKAN (Low order)
    - x_season: Đi vào rKAN (Mid order)
    - x_resid: Đi vào Jump Layer (để phát hiện Spikes)
    """
    def __init__(self, kernel_low=25, kernel_mid=11):
        super(CascadedDecomp, self).__init__()
        
        # Kernel lớn (vd: 25) để bắt xu hướng rất chậm
        self.decomp_low = SeriesDecomp(kernel_low)
        
        # Kernel nhỏ hơn (vd: 11) để bắt các dao động tuần hoàn
        self.decomp_mid = SeriesDecomp(kernel_mid)

    def forward(self, x):
        # Bước 1: Tách Trend
        # x_rest_1 chứa (Season + Noise + Jumps)
        # x_trend chứa (Trend)
        x_rest_1, x_trend = self.decomp_low(x)
        
        # Bước 2: Tách Season từ phần còn lại
        # x_resid chứa (Noise + Jumps) -> Đây là "thức ăn" cho Nhánh 2
        # x_season chứa (Season)
        x_resid, x_season = self.decomp_mid(x_rest_1)
        
        return x_trend, x_season, x_resid