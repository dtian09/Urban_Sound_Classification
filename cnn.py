import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from tqdm import tqdm

# Dataset class
class UrbanSoundDataset(Dataset):
    def __init__(self, csv_path, data_dir, fold, transform=None, sr=22050, duration=4, n_mels=64):
        self.metadata = pd.read_csv(csv_path)
        self.metadata = self.metadata[self.metadata['fold'] == fold]
        self.data_dir = data_dir
        self.transform = transform
        self.sr = sr
        self.target_len = sr * duration
        self.n_mels = n_mels

    def pad_or_truncate_waveform(self, y):#make sure each clip is 4 seconds (pad short clips to 4 s or truncate longer ones to 4s)
        if len(y) < self.target_len:
            y = np.pad(y, (0, self.target_len - len(y)))
        else:
            y = y[:self.target_len]
        return y

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.data_dir, f"fold{row['fold']}", row['slice_file_name'])
        label = row['classID']

        # Load and pad/truncate waveform
        waveform, sr = librosa.load(file_path, sr=self.sr)
        waveform = self.pad_or_truncate_waveform(waveform)

        # Compute Mel spectrogram
        melspec = librosa.feature.melspectrogram(y=waveform, sr=self.sr, n_mels=self.n_mels)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        melspec_db = np.expand_dims(melspec_db, axis=0)  # (1, n_mels, time)
      
        if self.transform:
            melspec_db = self.transform(melspec_db)

        return torch.tensor(melspec_db, dtype=torch.float32), label


# CNN Model
class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Compute the output size after conv and pool
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 173)  # (B, C, n_mels, time)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 16, 62, 171]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 32, 30, 84] → [B, 32, 15, 41]
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Training function
def train_model():
    csv_path = "/root/sound_datasets/urbansound8k/metadata/UrbanSound8K.csv"
    data_dir = "/root/sound_datasets/urbansound8k/audio"
    fold = 1

    dataset = UrbanSoundDataset(csv_path=csv_path, data_dir=data_dir, fold=fold)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = AudioCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}")

        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()
