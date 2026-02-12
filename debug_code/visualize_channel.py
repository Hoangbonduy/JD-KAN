import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
import shutil

# Thêm đường dẫn hiện tại
sys.path.append(os.getcwd())

from layers.DilatedJumpLayer import DilatedJumpLayer
from layers.StandardNorm import Normalize

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
class Config:
    def __init__(self):
        # Mặc định
        self.seq_len = 96
        self.pred_len = 96
        self.enc_in = 7
        self.d_model = 16  # Giữ 16 để vẽ 4x4 plot
        self.dropout = 0.1
        self.kan_order = 3
        self.e_layers = 2
        self.batch_size = 32 # Batch size cho training
        self.train_epochs = 5 # Số epoch demo (bạn có thể tăng lên)
        self.learning_rate = 0.001

    def load_from_sh(self, sh_path):
        if not os.path.exists(sh_path): 
            print(f"  ⚠ Không tìm thấy file {sh_path}, dùng config mặc định")
            return
        with open(sh_path, 'r', encoding='utf-8') as f:
            content = f.read()
        patterns = {
            'seq_len': r'--seq_len\s+(\d+)',
            'pred_len': r'--pred_len\s+(\d+)',
            'enc_in': r'--enc_in\s+(\d+)',
            'd_model': r'--d_model\s+(\d+)',
            'kan_order': r'--kan_order\s+(\d+)',
            'e_layers': r'--e_layers\s+(\d+)',
        }
        print("Đang đọc tham số từ scripts/run_jdkan_etth1.sh...")
        for name, pattern in patterns.items():
            match = re.search(pattern, content)
            if match: 
                setattr(self, name, int(match.group(1)))
                print(f"  - {name}: {match.group(1)}")
        
        # Warning nếu d_model != 16 (để vẽ 4x4 grid)
        if self.d_model != 16:
            print(f"  ⚠ Warning: d_model={self.d_model} (sẽ force về 16 để vẽ đẹp)")

# ==========================================
# 2. DATASET ĐƠN GIẢN
# ==========================================
class SimpleTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        # Input: [seq_len, features]
        # Target: Bài toán Reconstruction (tự tái tạo chính nó) để demo
        x = self.data[index : index + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)

# ==========================================
# 3. MODEL WRAPPER (Để train được)
# ==========================================
class TrainableJDKAN(nn.Module):
    def __init__(self, configs):
        super(TrainableJDKAN, self).__init__()
        self.configs = configs
        
        # Các thành phần y hệt code của bạn
        self.normalize = Normalize(configs.enc_in, affine=True, non_norm=False)
        self.enc_embedding = nn.Linear(configs.enc_in, configs.d_model)
        
        self.jump_layer = DilatedJumpLayer(
            d_model=configs.d_model,
            kernel_size=5,
            dilations=[1, 2, 4, 8, 16],
            kan_order=configs.kan_order,
            dropout=configs.dropout
        )
        
        # Thêm lớp output để tính Loss (đưa từ d_model 16 về lại enc_in 7)
        self.projection = nn.Linear(configs.d_model, configs.enc_in)

    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
        x_norm = self.normalize(x, 'norm')
        x_emb = self.enc_embedding(x_norm)
        
        # Dilated Layer logic
        x_out = self.jump_layer(x_emb) # Gọi hàm forward chuẩn của layer
        
        # Projection về lại kích thước gốc để tính loss
        output = self.projection(x_out)
        return output

# ==========================================
# 4. HÀM VISUALIZE (Chạy sau mỗi Epoch)
# ==========================================
def visualize_epoch(model, sample_input, epoch, save_dir):
    """
    Hàm này mổ xẻ model để vẽ lại đồ thị các lớp bên trong
    VỚI KIẾN TRÚC PARALLEL: Tất cả các conv nhìn vào cùng 1 input gốc
    """
    model.eval() # Chế độ đánh giá
    with torch.no_grad():
        # Chuẩn bị dữ liệu
        configs = model.configs
        x_norm = model.normalize(sample_input, 'norm')
        x_emb = model.enc_embedding(x_norm)
        
        # -- PARALLEL ARCHITECTURE: Tất cả conv nhận CÙNG input gốc --
        x_original = x_emb.permute(0, 2, 1) # [1, 16, 96] - Input gốc cho TẤT CẢ conv
        seq_len = x_original.size(2)
        
        layer_list = model.jump_layer.convs
        
        # Tạo folder cho epoch này
        epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
        if not os.path.exists(epoch_dir): os.makedirs(epoch_dir)

        print(f" >> Đang vẽ đồ thị cho Epoch {epoch} (PARALLEL MODE)...")
        
        # Lưu tất cả outputs để sau này concatenate (như model thực)
        all_outputs = []
        
        for i, conv in enumerate(layer_list):
            d = model.jump_layer.dilations[i]
            
            # QUAN TRỌNG: Mỗi conv nhận CÙNG input gốc, không phải output của conv trước
            x_in = x_original.clone()  # Luôn dùng input gốc
            x_out = conv(x_in)
            x_out = x_out[:, :, :seq_len] # Causal trim
            x_out = model.jump_layer.act(x_out)
            x_out = model.jump_layer.dropout(x_out)  # Dropout (tắt khi eval mode)
            
            # Lưu lại để concatenate
            all_outputs.append(x_out)
            
            # Vẽ và lưu ảnh: So sánh INPUT GỐC vs OUTPUT của conv này
            plot_and_save(x_in, x_out, d, i, epoch, epoch_dir)
        
        # Visualize concatenated output (trước khi qua fusion)
        concatenated = torch.cat(all_outputs, dim=1)  # [1, 16*5=80, 96]
        plot_concatenated(x_original, concatenated, epoch, epoch_dir)
        
        # Visualize sau fusion layer
        fused = model.jump_layer.fusion_layer(concatenated)  # [1, 16, 96]
        plot_fusion_output(x_original, fused, epoch, epoch_dir)

def plot_and_save(x_in, x_out, dilation, layer_idx, epoch, save_dir):
    data_in = x_in[0].cpu().numpy()
    data_out = x_out[0].cpu().numpy()
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    fig.suptitle(f'Epoch {epoch} - Layer {layer_idx} (Dilation {dilation}) - PARALLEL MODE\nInput: Original Data | Output: After Conv', fontsize=16)
    
    axes_flat = axes.flatten()
    for c in range(16):
        if c >= len(axes_flat): break
        ax = axes_flat[c]
        ax.plot(data_in[c], linestyle='--', alpha=0.4, color='gray', label='Input (Original)') # Input gốc
        ax.plot(data_out[c], linewidth=1.5, color='blue', label='Output') # Sau conv
        ax.set_title(f'Ch {c}', fontsize=8)
        if c < 12: ax.set_xticklabels([])
        if c % 4 != 0: ax.set_yticklabels([])
        ax.grid(alpha=0.3)
        if c == 0: ax.legend(fontsize=6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'layer_{layer_idx}_dil_{dilation}.png')
    plt.savefig(save_path)
    plt.close(fig) # Đóng hình để giải phóng RAM

def plot_concatenated(x_original, concatenated, epoch, save_dir):
    """Visualize concatenated multi-scale features"""
    data_orig = x_original[0].cpu().numpy()  # [16, 96]
    data_concat = concatenated[0].cpu().numpy()  # [80, 96] = 16 channels * 5 dilations
    
    # Plot mean của từng nhóm dilation
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Epoch {epoch} - Concatenated Multi-Scale Features (Before Fusion)', fontsize=16)
    
    # Plot 1: Original input (mean across all 16 channels)
    ax = axes[0, 0]
    ax.plot(data_orig.mean(axis=0), linewidth=2, color='black', label='Original (mean)')
    ax.set_title('Original Input (Mean)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2-6: Mean of each dilation group
    dilations = [1, 2, 4, 8, 16]
    for i, d in enumerate(dilations):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row, col]
        
        start_idx = i * 16
        end_idx = (i + 1) * 16
        group_data = data_concat[start_idx:end_idx, :]  # [16, 96]
        
        # Plot all channels in this dilation group
        for ch in range(16):
            ax.plot(group_data[ch, :], alpha=0.3, linewidth=0.5)
        # Plot mean
        ax.plot(group_data.mean(axis=0), linewidth=2, color='red', label='Mean')
        
        ax.set_title(f'Dilation {d} Group (16 channels)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, 'concatenated_features.png')
    plt.savefig(save_path)
    plt.close(fig)

def plot_fusion_output(x_original, fused, epoch, save_dir):
    """Visualize output after fusion layer (1x1 conv that reduces 80 channels back to 16)"""
    data_orig = x_original[0].cpu().numpy()  # [16, 96]
    data_fused = fused[0].cpu().numpy()  # [16, 96]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    fig.suptitle(f'Epoch {epoch} - After Fusion Layer (1x1 Conv: 80→16 channels)\nGray=Original | Blue=After Fusion', fontsize=16)
    
    axes_flat = axes.flatten()
    for c in range(16):
        if c >= len(axes_flat): break
        ax = axes_flat[c]
        ax.plot(data_orig[c], linestyle='--', alpha=0.4, color='gray', label='Original')
        ax.plot(data_fused[c], linewidth=1.5, color='blue', label='Fused')
        ax.set_title(f'Ch {c}', fontsize=8)
        if c < 12: ax.set_xticklabels([])
        if c % 4 != 0: ax.set_yticklabels([])
        ax.grid(alpha=0.3)
        if c == 0: ax.legend(fontsize=6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, 'after_fusion.png')
    plt.savefig(save_path)
    plt.close(fig)

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
def main():
    # Setup
    configs = Config()
    configs.load_from_sh('scripts/run_jdkan_etth1.sh')
    # Force d_model = 16 để vẽ đẹp (nếu file sh để 512 thì ghi đè lại ở đây)
    configs.d_model = 16 
    
    # Path
    save_dir = 'training_visuals'
    if os.path.exists(save_dir): shutil.rmtree(save_dir) # Xóa cũ
    os.makedirs(save_dir)

    # Load Data
    csv_path = 'dataset/ETT-small/ETTh1.csv'
    if not os.path.exists(csv_path): csv_path = 'ETTh1.csv'
    
    if os.path.exists(csv_path):
        print(f"Load dữ liệu: {csv_path}")
        df = pd.read_csv(csv_path)
        data_values = df.iloc[:, 1:].values # Bỏ cột date
        data_values = data_values[:, :configs.enc_in]
        
        # Chuẩn hóa sơ bộ (StandardScaler)
        mean = np.mean(data_values, axis=0)
        std = np.std(data_values, axis=0)
        data_values = (data_values - mean) / std
    else:
        print("Dùng dữ liệu random để test code.")
        data_values = np.random.randn(500, configs.enc_in)

    # Dataset & DataLoader
    dataset = SimpleTimeSeriesDataset(data_values, configs.seq_len)
    dataloader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)
    
    # Lấy 1 mẫu cố định để theo dõi xuyên suốt quá trình train
    sample_input, _ = dataset[0]
    sample_input = sample_input.unsqueeze(0) # [1, seq_len, 7]

    # Model & Optimizer
    model = TrainableJDKAN(configs)
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    criterion = nn.MSELoss()

    print(f"\n--- BẮT ĐẦU TRAINING ({configs.train_epochs} Epochs) ---")
    print(f"Ảnh kết quả sẽ được lưu tại thư mục: {save_dir}/")

    # --- Training Loop ---
    for epoch in range(1, configs.train_epochs + 1):
        model.train()
        total_loss = 0
        
        for i, (batch_x, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward
            outputs = model(batch_x)
            
            # Loss (Reconstruction)
            loss = criterion(outputs, batch_y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{configs.train_epochs} | Loss: {avg_loss:.6f}")
        
        # --- VISUALIZE SAU MỖI EPOCH ---
        visualize_epoch(model, sample_input, epoch, save_dir)

    print("\nTraining hoàn tất!")
    print("Hãy mở thư mục 'training_visuals' để xem mô hình học thế nào qua từng Epoch.")

if __name__ == "__main__":
    main()