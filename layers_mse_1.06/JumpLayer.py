import torch
import torch.nn as nn
import torch.nn.functional as F

class JumpLayer(nn.Module):
    """
    Module 3: Sparse Gated Jump Layer
    Input: Residual component [Batch, Seq_Len, Channels]
    Output: Predicted Future Jumps [Batch, Pred_Len, Channels]
    
    Novelty: 
    - Uses a 'Gating Mechanism' to enforce sparsity.
    - Unlike trend, jumps are rare events. This layer learns 
      to be 'silent' most of the time and only activates when 
      significant shocks are detected.
    """
    def __init__(self, seq_len, pred_len, n_channels, d_model=32, dropout=0.1):
        super(JumpLayer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        
        # 1. Anomaly Feature Extractor (CNN)
        # Captures local spikes/abrupt changes in the residual
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(negative_slope=0.2), # Leaky ReLU good for anomalies
            nn.Dropout(dropout)
        )
        
        # 2. Dual-Head Prediction Mechanism
        # Instead of predicting values directly, we split into "Where" and "How much"
        
        # Head A: The 'Gate' (Where/Probability of Jump)
        # Projects time axis S -> P
        self.gate_projector = nn.Linear(seq_len, pred_len)
        self.gate_out = nn.Linear(d_model, n_channels)
        
        # Head B: The 'Value' (Magnitude of Jump)
        # Projects time axis S -> P
        self.val_projector = nn.Linear(seq_len, pred_len)
        self.val_out = nn.Linear(d_model, n_channels)
        
    def forward(self, x_residual):
        """
        x_residual: [Batch, Seq_Len, Channels]
        """
        B, S, C = x_residual.shape
        
        # --- Step 1: Feature Extraction ---
        # Permute for Conv1d: [B, C, S]
        x_in = x_residual.permute(0, 2, 1)
        
        # Extract features: [B, d_model, S]
        feat = self.conv_in(x_in)
        
        # --- Step 2: Time Projection ---
        # Need to project S -> P for both heads
        # Permute to [B, d_model, S] -> [B, d_model, P] via Linear on last dim?
        # No, Linear operates on last dim. We need to apply it on Time axis.
        
        # Transpose to [B, d_model, S]
        # We want to project S -> P. 
        # So we treat d_model as batch/features effectively.
        
        # Method: Use Linear on Time Axis
        # Input to Linear: [B, d_model, S] -> Output: [B, d_model, P]
        feat_gate = self.gate_projector(feat) # [B, d_model, P]
        feat_val = self.val_projector(feat)   # [B, d_model, P]
        
        # --- Step 3: Dual Heads ---
        # Reshape to apply Linear on d_model dim: [B, P, d_model]
        feat_gate = feat_gate.permute(0, 2, 1)
        feat_val = feat_val.permute(0, 2, 1)
        
        # A. Compute Gate (Probability mask)
        # Sigmoid forces output to [0, 1]
        # A hard threshold (e.g., < 0.1 becomes 0) can be applied during inference if needed
        gate = torch.sigmoid(self.gate_out(feat_gate)) # [B, P, C]
        
        # B. Compute Value (Magnitude)
        value = self.val_out(feat_val) # [B, P, C]
        
        # --- Step 4: Gated Fusion ---
        # Output is Value weighted by Probability
        # If gate is close to 0, output is 0 (Sparse)
        future_jumps = value * gate
        
        return future_jumps