import torch
import torch.nn as nn
import torch.nn.functional as F

class LightCrossAttention(nn.Module):
    """
    Lightweight Cross-Attention Mechanism.
    Purpose: Allows the Continuous Trend (Query) to attend to the Jumps (Key/Value).
    Meaning: "Adjust the trend prediction based on where the shocks are located."
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super(LightCrossAttention, self).__init__()
        
        # Multi-head Attention
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_query, x_kv):
        """
        x_query: Trend component [Batch, Seq_Len, d_model]
        x_kv: Jump component [Batch, Seq_Len, d_model]
        """
        # Attention(Q, K, V)
        # attn_output shape: [Batch, Seq_Len, d_model]
        attn_output, _ = self.mha(query=x_query, key=x_kv, value=x_kv)
        
        # Residual Connection + Norm
        output = self.norm(x_query + self.dropout(attn_output))
        return output

class FusionLayer(nn.Module):
    """
    Module 4: Adaptive Fusion Layer
    Combines:
    1. Continuous Trend (from DiffusionLayer)
    2. Discrete Jumps (from JumpLayer)
    
    Mechanism: 
    - Cross-Attention to refine Trend based on Jumps.
    - Adaptive Mixing Weight to combine them.
    """
    def __init__(self, n_channels, d_model=64):
        super(FusionLayer, self).__init__()
        
        # Project inputs to d_model for Attention (if they are not already)
        # Assuming inputs are [Batch, Pred_Len, Channels], we need to project to d_model
        self.trend_proj = nn.Linear(n_channels, d_model)
        self.jump_proj = nn.Linear(n_channels, d_model)
        
        # The Interaction Mechanism
        self.cross_attn = LightCrossAttention(d_model=d_model, n_heads=4)
        
        # Project back to Channels
        self.out_proj = nn.Linear(d_model, n_channels)
        
        # Adaptive Mixing Parameter (Learnable Scalar per channel)
        # Initialize alpha=0 -> tanh(0)=0 -> impact=1 (Standard addition initially)
        self.alpha = nn.Parameter(torch.zeros(1, 1, n_channels))
        
        # Final Smoothing
        self.dropout = nn.Dropout(0.1)

    def forward(self, future_trend, future_jumps):
        """
        Inputs:
            future_trend: [Batch, Pred_Len, Channels]
            future_jumps: [Batch, Pred_Len, Channels]
        Output:
            final_prediction: [Batch, Pred_Len, Channels]
        """
        # 1. Projection to Feature Space (d_model) for Attention
        # [B, P, C] -> [B, P, d_model]
        trend_feat = self.trend_proj(future_trend)
        jump_feat = self.jump_proj(future_jumps)
        
        # 2. Interaction (Refine Trend based on Jumps)
        # Query = Trend, Key/Value = Jump
        # "How should the trend curve bend given these upcoming shocks?"
        trend_refined_feat = self.cross_attn(trend_feat, jump_feat)
        
        # 3. Projection back to Signal Space
        # [B, P, d_model] -> [B, P, C]
        trend_refined = self.out_proj(trend_refined_feat)
        
        # 4. Adaptive Mixing
        # Formula: y = Trend_Refined + (1 + tanh(alpha)) * Jump
        # tanh(alpha) ranges [-1, 1], so impact ranges [0, 2]
        # This allows the model to amplify or dampen the jump's impact dynamically.
        jump_impact = 1.0 + torch.tanh(self.alpha)
        
        final_prediction = trend_refined + (jump_impact * future_jumps)
        
        return self.dropout(final_prediction)