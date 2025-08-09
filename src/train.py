import sys
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import RagaDataset
from models.cnn_backbone import CNNBackbone
from models.kan import KANHead
from models.proto_head import ProtoNetHead
from tqdm import tqdm
import numpy as np

# Define constants
SAMPLE_RATE = 22050
DURATION = 5.0  # seconds
HOP_LENGTH = 512  # for CFCC feature extraction
MAX_TIME_FRAMES = int(np.ceil(DURATION * SAMPLE_RATE / HOP_LENGTH))  # Calculate max time frames

# Config
DATA_ROOT = '../PallaviData'  # Path relative to src directory
N_EPOCHS = 100  # Increased epochs for better training
N_CHROMA = 12  # Number of chroma bins
N_CFCC = 13    # Number of CFCC coefficients
FEAT_DIM = 128
LEARNING_RATE = 1e-4  # Reduced learning rate for more stable training

# Few-shot learning parameters
N_WAY = 5  # Number of classes per episode (for training stability)
N_SHOT = 2  # Number of support examples per class
N_QUERY = 3  # Number of query examples per class
EPISODES_PER_EPOCH = 200  # More episodes per epoch for better convergence

# Create output directory for saving models
os.makedirs('../checkpoints', exist_ok=True)

# Dataset
train_set = RagaDataset(DATA_ROOT, augment=True, n_chroma=N_CHROMA, n_cfcc=N_CFCC, sr=SAMPLE_RATE, duration=DURATION, max_time_frames=MAX_TIME_FRAMES)
val_set = RagaDataset(DATA_ROOT, augment=False, n_chroma=N_CHROMA, n_cfcc=N_CFCC, sr=SAMPLE_RATE, duration=DURATION, max_time_frames=MAX_TIME_FRAMES)

print(f"Total number of raga classes in dataset: {len(train_set.ragas)}")
print(f"Training with {N_WAY}-way {N_SHOT}-shot episodes")
print(f"Available ragas: {list(train_set.ragas.keys())}")

# Model - CFCC features have dimensions (min(n_cfcc, n_chroma) + n_chroma, time_frames)
# Since DCT can only produce as many coefficients as input features (12 chroma bins)
cfcc_height = N_CHROMA + min(N_CFCC, N_CHROMA)  # 12 + min(13, 12) = 12 + 12 = 24
cnn = CNNBackbone(out_dim=FEAT_DIM, input_height=cfcc_height, input_width=MAX_TIME_FRAMES)
kan = KANHead(in_dim=FEAT_DIM, hidden_dim=128, out_dim=FEAT_DIM)  # Increased hidden dim for better capacity
proto = ProtoNetHead()

# Optimizer with improved settings
optimizer = torch.optim.AdamW(  # Use AdamW for better regularization
    list(cnn.parameters()) + list(kan.parameters()), 
    lr=LEARNING_RATE,
    weight_decay=1e-4  # Increased weight decay for better regularization
)

# Learning rate scheduler with cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn.to(device)
kan.to(device)
proto.to(device)

print(f"Training on {device}")

# Training loop
for epoch in range(N_EPOCHS):
    cnn.train()
    kan.train()
    
    epoch_losses = []
    epoch_accs = []
    
# Training loop
best_val_acc = 0.0
patience = 20
patience_counter = 0

for epoch in range(N_EPOCHS):
    cnn.train()
    kan.train()
    
    epoch_losses = []
    epoch_accs = []
    
    # Training episodes
    for episode in tqdm(range(EPISODES_PER_EPOCH), desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]"):
        try:
            # Generate a few-shot episode
            (support_data, support_labels), (query_data, query_labels) = train_set.get_episode(
                n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY
            )
            
            # Skip if not enough data
            if len(support_data) < N_WAY * N_SHOT or len(query_data) < N_WAY * N_QUERY:
                continue
                
            # Move to device
            support_data = support_data.to(device)  # (n_way*n_shot, cfcc_height, time)
            support_labels = support_labels.to(device)  # (n_way*n_shot,)
            query_data = query_data.to(device)  # (n_way*n_query, cfcc_height, time)
            query_labels = query_labels.to(device)  # (n_way*n_query,)
            
            # Add channel dimension for CNN
            support_data = support_data.unsqueeze(1)  # (n_way*n_shot, 1, cfcc_height, time)
            query_data = query_data.unsqueeze(1)  # (n_way*n_query, 1, cfcc_height, time)
            
            # Extract features
            support_feats = cnn(support_data)  # (n_way*n_shot, feat_dim)
            query_feats = cnn(query_data)  # (n_way*n_query, feat_dim)
            
            # Apply KAN transformation (this is the key improvement)
            support_feats = kan(support_feats)  # (n_way*n_shot, feat_dim)
            query_feats = kan(query_feats)  # (n_way*n_query, feat_dim)
            
            # Prototypical network classification
            logits = proto(support_feats, support_labels, query_feats)  # (n_way*n_query, n_way)
            
            # Compute loss
            loss = F.cross_entropy(logits, query_labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(list(cnn.parameters()) + list(kan.parameters()), max_norm=1.0)
            
            optimizer.step()
            
            # Compute accuracy
            preds = logits.argmax(dim=1)
            acc = (preds == query_labels).float().mean()
            
            epoch_losses.append(loss.item())
            epoch_accs.append(acc.item())
            
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            continue
    
    # Validation
    cnn.eval()
    kan.eval()
    val_losses = []
    val_accs = []
    
    with torch.no_grad():
        for episode in tqdm(range(50), desc=f"Epoch {epoch+1}/{N_EPOCHS} [Val]"):  # Fewer validation episodes
            try:
                (support_data, support_labels), (query_data, query_labels) = val_set.get_episode(
                    n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY
                )
                
                if len(support_data) < N_WAY * N_SHOT or len(query_data) < N_WAY * N_QUERY:
                    continue
                
                support_data = support_data.to(device).unsqueeze(1)
                support_labels = support_labels.to(device)
                query_data = query_data.to(device).unsqueeze(1)
                query_labels = query_labels.to(device)
                
                support_feats = kan(cnn(support_data))
                query_feats = kan(cnn(query_data))
                logits = proto(support_feats, support_labels, query_feats)
                
                val_loss = F.cross_entropy(logits, query_labels)
                val_acc = (logits.argmax(dim=1) == query_labels).float().mean()
                
                val_losses.append(val_loss.item())
                val_accs.append(val_acc.item())
                
            except Exception as e:
                continue
    
    # Update learning rate
    scheduler.step()
    
    # Print epoch statistics
    avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
    avg_acc = np.mean(epoch_accs) if epoch_accs else 0.0
    val_avg_loss = np.mean(val_losses) if val_losses else float('inf')
    val_avg_acc = np.mean(val_accs) if val_accs else 0.0
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, "
          f"Val Loss={val_avg_loss:.4f}, Val Acc={val_avg_acc:.4f}, LR={current_lr:.6f}")
    
    # Early stopping and model saving
    if val_avg_acc > best_val_acc:
        best_val_acc = val_avg_acc
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'cnn_state_dict': cnn.state_dict(),
            'kan_state_dict': kan.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'train_loss': avg_loss,
            'val_loss': val_avg_loss,
        }, '../checkpoints/model_best.pth')
        print(f"New best validation accuracy: {best_val_acc:.4f}")
    else:
        patience_counter += 1
        
    # Save checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch,
            'cnn_state_dict': cnn.state_dict(),
            'kan_state_dict': kan.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': val_avg_loss,
        }, f'../checkpoints/model_epoch_{epoch+1}.pth')
    
    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Save final model
torch.save({
    'cnn_state_dict': cnn.state_dict(),
    'kan_state_dict': kan.state_dict(),
    'final_epoch': epoch,
    'best_val_acc': best_val_acc,
}, '../checkpoints/model_epoch_last.pth')

print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
print("Models saved:")
print("- ../checkpoints/model_best.pth (best validation accuracy)")
print("- ../checkpoints/model_epoch_last.pth (final model)")
