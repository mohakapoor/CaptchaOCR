import torch
import torch.nn as nn
from src.config import cfg

class SmallCNN(nn.Module):
    """
    Improved CNN with BatchNorm and residual connections.
    Produces feature map with total stride 4 along width,
    and compresses height to ~1 via pooling.
    """
    def __init__(self, in_ch=1) -> None:
        super().__init__()
        # First conv block: H,W -> H/2, W/2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))  # stride 2x2
        )
        
        # Second conv block: maintain H/2, W/2 -> W/4
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))  # height stride 1, width stride 2
        )
        
        # Residual block at 128 channels
        self.residual = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )
        self.residual_relu = nn.ReLU(inplace=True)
        
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))  # squeeze height to 1

    def forward(self, x):
        # First two conv blocks
        f = self.conv1(x)           # [B, 64, H/2, W/2]
        f = self.conv2(f)           # [B, 128, H/2, W/4]
        
        # Residual connection
        residual = f
        f = self.residual(f)        # [B, 128, H/2, W/4]
        f = f + residual            # Skip connection
        f = self.residual_relu(f)   # [B, 128, H/2, W/4]
        
        # Height pooling
        f = self.height_pool(f)     # [B, 128, 1, W/4]
        f = f.squeeze(2)            # [B, 128, W/4]
        f = f.permute(2, 0, 1)      # [T(=W/4), B, 128]
        return f

class CRNN(nn.Module):
    def __init__(self, vocab_size: int, in_ch: int = 1, hidden: int = 320, layers: int = 2, dropout: float = 0.05):
        super().__init__()
        self.cnn = SmallCNN(in_ch=in_ch)
        self.rnn = nn.LSTM(input_size=128, hidden_size=hidden, num_layers=layers,
                           bidirectional=True, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(2*hidden)  # Add LayerNorm for stability
        self.fc = nn.Linear(2*hidden, vocab_size)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        # Initialize final linear layer with small weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        seq = self.cnn(x)   # [T,B,C=128]
        y, _ = self.rnn(seq)  # [T,B,2H]
        y = self.norm(y)     # [T,B,2H] - Apply LayerNorm
        logits = self.fc(y)   # [T,B,V]
        return logits