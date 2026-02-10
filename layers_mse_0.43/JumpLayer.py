import torch
import torch.nn as nn
import torch.nn.functional as F

class JumpLayer(nn.Module):
    """
    Nhánh 2: Discrete Jump Stream (Luồng Nhảy Vọt Rời Rạc)
    Sử dụng cơ chế Memory Bank để "nhớ" và "gọi lại" các mẫu hình sốc (shocks/jumps).
    """
    def __init__(self, input_dim, seq_len, pred_len, n_jumps=32, d_model=64):
        super(JumpLayer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_jumps = n_jumps
        
        # 1. Gating Network (Trigger)
        # Quyết định cường độ (Intensity) của Jump dựa trên Residual quá khứ
        # Input: [Batch, Channels, Seq_Len] -> Output: [Batch, Channels, 1]
        self.gating_net = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid() # Output xác suất [0, 1]
        )
        
        # 2. Event Memory Bank
        # Lưu trữ các mẫu hình Jump (Profile) cho tương lai
        # Shape: [N_Jumps, Pred_Len]
        # Khởi tạo nhỏ để không gây nhiễu lúc đầu
        self.memory_bank = nn.Parameter(torch.randn(n_jumps, pred_len) * 0.02)
        
        # 3. Retrieval Query Projector
        # Chiếu Residual quá khứ thành vector Query để so khớp với Memory
        # Map: Seq_Len -> N_Jumps (Logits chọn jump nào)
        self.query_proj = nn.Linear(seq_len, n_jumps)

    def forward(self, x_resid):
        # x_resid: [Batch, Seq_Len, Channels]
        # Transpose để xử lý chiều thời gian: [Batch, Channels, Seq_Len]
        x_resid_t = x_resid.permute(0, 2, 1)
        
        # --- A. Gating Mechanism (When & How much?) ---
        # Tính cường độ jump cho từng channel
        # gate: [Batch, Channels, 1]
        gate = self.gating_net(x_resid_t)
        
        # Thresholding (Optional): Có thể thêm logic if gate < 0.1 then 0
        # Nhưng để train được (differentiable), ta giữ nguyên giá trị liên tục
        
        # --- B. Retrieval Mechanism (Which pattern?) ---
        # Tính điểm số phù hợp (Matching Score)
        # logits: [Batch, Channels, N_Jumps]
        retrieval_logits = self.query_proj(x_resid_t)
        
        # Chuyển thành trọng số (Attention Weights)
        # weights: [Batch, Channels, N_Jumps]
        retrieval_weights = F.softmax(retrieval_logits, dim=-1)
        
        # Lấy tổ hợp các mẫu jump từ Memory Bank
        # Công thức: Sum(Weights * Memory_Patterns)
        # [Batch, Channels, N_Jumps] x [N_Jumps, Pred_Len] -> [Batch, Channels, Pred_Len]
        selected_jump = torch.matmul(retrieval_weights, self.memory_bank)
        
        # --- C. Final Output ---
        # Jump = Gate * Pattern
        # Nếu Gate ~ 0 (không có shock), Jump sẽ tắt.
        future_jumps = gate * selected_jump
        
        # Transpose về lại [Batch, Pred_Len, Channels] để khớp với model chính
        return future_jumps.permute(0, 2, 1)