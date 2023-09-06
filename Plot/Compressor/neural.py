import torch.nn as nn
from torch.utils.data import Dataset


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            # nn.Linear(64, 64),
            # nn.GELU(),
            nn.Linear(64, 2),  # Output layer, 64 dimensions to 2 dimensions
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            # nn.Linear(64, 64),
            # nn.GELU(),
            nn.Linear(64, 64),  # Output layer, 64 dimensions to 2 dimensions
        )

    def forward(self, x):
        return self.model(x)

class Scorer(nn.Module):
    def __init__(self):
        super(Scorer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            # nn.Linear(64, 64),
            # nn.GELU(),
            nn.Linear(64, 1),  # Final output layer, 64 dimensions to 1 dimension
        )

    def forward(self, x):
        return self.model(x)
    


class CustomDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]
        return input_sample, target_sample