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

class MultiScaleDecomp(nn.Module):
    """
    Phân rã đa tỉ lệ:
    - Kernel nhỏ (3): Bắt nhiễu và sự kiện đột ngột (cho nhánh Jump).
    - Kernel vừa (13): Bắt tính chu kỳ (cho nhánh Season).
    - Kernel lớn (33): Bắt xu hướng dài hạn (cho nhánh Trend).
    """
    def __init__(self, kernel_sizes=[3, 13, 33]):
        super(MultiScaleDecomp, self).__init__()
        # Sắp xếp kernel từ nhỏ đến lớn để dễ xử lý
        self.kernel_sizes = sorted(kernel_sizes)
        self.decomps = nn.ModuleList([SeriesDecomp(k) for k in self.kernel_sizes])

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        
        # 1. Lấy Trend từ kernel lớn nhất (mượt nhất)
        _, trend_feat = self.decomps[-1](x) # Kernel 33
        
        # 2. Lấy Residual tổng thể từ kernel nhỏ nhất (chi tiết nhất)
        # resid_feat ở đây chứa tất cả những gì kernel 3 không bắt được (tức là dao động cực nhanh)
        resid_feat, _ = self.decomps[0](x)  # Kernel 3
        
        # 3. Tính Season
        # Season là phần nằm giữa: Không phải Trend dài hạn, cũng không phải nhiễu cực nhanh
        # Công thức: Season = Original - Trend - Resid
        season_feat = x - trend_feat - resid_feat
        
        return trend_feat, season_feat, resid_feat