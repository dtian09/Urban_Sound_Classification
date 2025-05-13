from cnn import UrbanSoundDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))  # learnable

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with residual and norm
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual and norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class AudioTransformer(nn.Module):
    def __init__(self, input_dim, conv_dim, max_len, n_layers, n_heads, num_classes):
        super().__init__()
        # 1D Convolution to extract local features
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=3, padding=1)
    
        # Transformer input projection
        self.input_proj = nn.Linear(conv_dim, conv_dim)
        self.pos_encoder = PositionalEncoding(max_len, conv_dim)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(conv_dim, n_heads) for _ in range(n_layers)
        ])
        
        # Final linear classification layer
        self.classifier = nn.Linear(conv_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, time, input_dim) - e.g. MFCCs
        """
        x = x.transpose(1, 2)                       # (B, C_in, T)
        x = self.conv1d(x)                          # (B, conv_dim, T)
        x = x.transpose(1, 2)                       # (B, T, conv_dim)
        x = self.input_proj(x)                      # (B, T, d_model)
        x = self.pos_encoder(x)                     # Add position encoding

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=1)                           # Global average pooling
        return self.classifier(x)                   # (B, num_classes)

from tqdm import tqdm
import numpy as np

def compute_global_mean_std(dataset):
    all_specs = []

    for spec, _ in tqdm(dataset):
        all_specs.append(spec.numpy())

    all_specs = np.concatenate(all_specs, axis=0)  # (total_time, n_mels)
    mean = all_specs.mean(axis=0)                 # (n_mels,)
    std = all_specs.std(axis=0) + 1e-6            # (n_mels,)
    return mean, std


dataset = UrbanSoundDataset(
    csv_path = "/root/sound_datasets/urbansound8k/metadata/UrbanSound8K.csv",
    data_dir = "/root/sound_datasets/urbansound8k/audio",
    fold=1,  # change for each fold
    transform=None,  # already returns dB mel spec
    sr=22050,
    duration=4,
    n_mels=64
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size=16

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

_, _, time = next(iter(train_loader))[0].squeeze(1).shape

model = AudioTransformer(
    input_dim=64,       # n_mels
    conv_dim=128,
    max_len=time,        # you can compute this from dataset: time frames from 4s audio
    n_layers=4,
    n_heads=4,
    num_classes=10
)

import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    total_loss, total_correct = 0, 0

    for batch in train_loader:
        x, y = batch
        x = x.squeeze(1).transpose(1, 2).to(device)  # shape: (B, T, F)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()

    acc = total_correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}: Loss={total_loss:.2f}, Acc={acc:.4f}")

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            #inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = batch
            inputs = inputs.squeeze(1).transpose(1, 2).to(device)  # shape: (B, T, F)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

