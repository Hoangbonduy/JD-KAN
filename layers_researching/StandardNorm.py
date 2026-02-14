import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            
            # --- [SỬA LỖI QUAN TRỌNG] ---
            # Tránh chia cho 0 khi affine_weight quá nhỏ
            # Thay vì: x = x / self.affine_weight
            weight = self.affine_weight
            # Giữ nguyên dấu, nhưng đảm bảo độ lớn tối thiểu là 1e-4
            safe_weight = torch.where(
                torch.abs(weight) < 1e-4, 
                torch.sign(weight) * 1e-4, 
                weight
            )
            # Xử lý trường hợp weight = 0 tuyệt đối (sign=0)
            safe_weight = torch.where(safe_weight == 0, torch.tensor(1e-4).to(x.device), safe_weight)
            
            x = x / safe_weight
            # ----------------------------
            
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x