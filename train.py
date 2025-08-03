import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.dataset import RagaDataset
from src.models.cnn_backbone import CNNBackbone
from src.models.kan import KANHead
from src.models.proto_head import ProtoNetHead
from tqdm import tqdm
import numpy as np
import os

# Define constants
SAMPLE_RATE = 22050
DURATION = 5.0  # seconds
HOP_LENGTH = 512  # for melspec
MAX_TIME_FRAMES = int(np.ceil(DURATION * SAMPLE_RATE / HOP_LENGTH))  # Calculate max time frames

# Config
DATA_ROOT = 'PallaviData'  # or your dataset path
N_EPOCHS = 50
N_MELS = 128
FEAT_DIM = 128
# Determine the total number of labels in the dataset
train_set = RagaDataset(DATA_ROOT, augment=True, n_mels=N_MELS, sr=SAMPLE_RATE, duration=DURATION, max_time_frames=MAX_TIME_FRAMES)
N_WAY = len(train_set.ragas)  # Total number of labels
N_SHOT = 2
N_QUERY = 2

# Create output directory for saving models
os.makedirs('checkpoints', exist_ok=True)

# Dataset
train_set = RagaDataset(DATA_ROOT, augment=True, n_mels=N_MELS, sr=SAMPLE_RATE, duration=DURATION, max_time_frames=MAX_TIME_FRAMES)

# Model
cnn = CNNBackbone(out_dim=FEAT_DIM, input_height=N_MELS, input_width=MAX_TIME_FRAMES)
kan = KANHead(in_dim=FEAT_DIM, hidden_dim=64, out_dim=FEAT_DIM)
proto = ProtoNetHead()

# Optimizer
optimizer = torch.optim.Adam(
    list(cnn.parameters()) + list(kan.parameters()), 
    lr=1e-3,
    weight_decay=1e-5
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

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
    
    # Generate multiple episodes per epoch
    for episode in tqdm(range(100), desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
        # Generate a few-shot episode
        (support_data, support_labels), (query_data, query_labels) = train_set.get_episode(
            n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY
        )
        
        # Move to device
        support_data = support_data.to(device)  # (n_way*n_shot, n_mels, time)
        support_labels = support_labels.to(device)  # (n_way*n_shot,)
        query_data = query_data.to(device)  # (n_way*n_query, n_mels, time)
        query_labels = query_labels.to(device)  # (n_way*n_query,)
        
        # Add channel dimension for CNN
        support_data = support_data.unsqueeze(1)  # (n_way*n_shot, 1, n_mels, time)
        query_data = query_data.unsqueeze(1)  # (n_way*n_query, 1, n_mels, time)
        
        # Extract features
        support_feats = cnn(support_data)  # (n_way*n_shot, feat_dim)
        query_feats = cnn(query_data)  # (n_way*n_query, feat_dim)
        
        # Apply KAN transformation
        support_feats = kan(support_feats)  # (n_way*n_shot, feat_dim)
        query_feats = kan(query_feats)  # (n_way*n_query, feat_dim)
        
        # Prototypical network classification
        logits = proto(support_feats, support_labels, query_feats)  # (n_way*n_query, n_way)
        
        # Compute loss
        loss = F.cross_entropy(logits, query_labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == query_labels).float().mean()
        
        epoch_losses.append(loss.item())
        epoch_accs.append(acc.item())
    
    # Update learning rate
    scheduler.step()
    
    # Print epoch statistics
    avg_loss = np.mean(epoch_losses)
    avg_acc = np.mean(epoch_accs)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
    
    # Save model checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'cnn_state_dict': cnn.state_dict(),
            'kan_state_dict': kan.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/model_epoch_{epoch+1}.pth')

# Save final model
torch.save({
    'cnn_state_dict': cnn.state_dict(),
    'kan_state_dict': kan.state_dict(),
}, 'checkpoints/final_model.pth')

print("Training completed. Model saved to 'checkpoints/final_model.pth'")
