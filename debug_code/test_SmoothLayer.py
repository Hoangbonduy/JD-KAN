import sys
sys.path.append('/home/hoang/experiments/JD-KAN')

import torch
import numpy as np
import matplotlib.pyplot as plt
from layers.SmoothLayer import SmoothLayer
import pandas as pd

def load_real_data():
    """Load real ETTh1 data"""
    try:
        df = pd.read_csv('/home/hoang/experiments/JD-KAN/dataset/ETT-small/ETTh1.csv')
        # Get all data points, select OT column (target)
        data = df['OT'].values
        print(f"Loaded {len(data)} data points from ETTh1.csv")
        return data
    except Exception as e:
        print(f"Could not load ETTh1 data: {e}")
        print("Using synthetic data instead")
        return None

def test_smooth_layer():
    """Test SmoothLayer and visualize results"""
    
    # Load or create data
    data = load_real_data()
    seq_len = len(data)
    
    # Prepare input tensor [Batch=1, Seq_Len, Channels=1]
    x_raw = torch.FloatTensor(data).view(1, seq_len, 1)
    
    # Initialize SmoothLayer
    smooth_layer = SmoothLayer(
        n_channels=1, 
        seq_len=seq_len,
        n_fourier_terms=8,
        d_model=32
    )
    
    # Forward pass
    print("Running SmoothLayer forward pass...")
    with torch.no_grad():
        x_smooth, residual = smooth_layer(x_raw)
    
    # Convert to numpy for plotting
    x_raw_np = x_raw[0, :, 0].numpy()
    x_smooth_np = x_smooth[0, :, 0].numpy()
    residual_np = residual[0, :, 0].numpy()
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    time_steps = np.arange(seq_len)
    
    # Plot 1: Original time series
    axes[0].plot(time_steps, x_raw_np, 'b-', linewidth=1.5, alpha=0.7, label='Original')
    axes[0].set_title('Original Time Series', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Smooth component
    axes[1].plot(time_steps, x_raw_np, 'b-', linewidth=1, alpha=0.3, label='Original')
    axes[1].plot(time_steps, x_smooth_np, 'r-', linewidth=2, label='Smooth (Trend)')
    axes[1].set_title('Smooth Component (x_smooth) - Extracted Trend', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Residual component
    axes[2].plot(time_steps, residual_np, 'g-', linewidth=1, alpha=0.7, label='Residual (Noise + Jumps)')
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[2].set_title('Residual Component - High-Frequency Noise & Anomalies', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/hoang/experiments/JD-KAN/debug_plots/smooth_layer_decomposition.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Original Signal:")
    print(f"  Mean: {x_raw_np.mean():.4f}")
    print(f"  Std:  {x_raw_np.std():.4f}")
    print(f"  Min:  {x_raw_np.min():.4f}")
    print(f"  Max:  {x_raw_np.max():.4f}")
    print(f"\nSmooth Component:")
    print(f"  Mean: {x_smooth_np.mean():.4f}")
    print(f"  Std:  {x_smooth_np.std():.4f}")
    print(f"  Min:  {x_smooth_np.min():.4f}")
    print(f"  Max:  {x_smooth_np.max():.4f}")
    print(f"\nResidual Component:")
    print(f"  Mean: {residual_np.mean():.4f}")
    print(f"  Std:  {residual_np.std():.4f}")
    print(f"  Min:  {residual_np.min():.4f}")
    print(f"  Max:  {residual_np.max():.4f}")
    print(f"\nReconstruction Error:")
    reconstruction = x_smooth_np + residual_np
    error = np.abs(x_raw_np - reconstruction).mean()
    print(f"  MAE: {error:.6f}")
    print("="*60)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs('/home/hoang/experiments/JD-KAN/debug_plots', exist_ok=True)
    
    print("Testing SmoothLayer...")
    test_smooth_layer()
