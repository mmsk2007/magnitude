import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MagNetAttention(nn.Module):
    """
    MagNet v2: Advanced Transformer-based architecture for scalar price displacement.
    Uses multi-head attention to capture long-term energy dependencies.
    """
    def __init__(self, input_dim=10, hidden_dim=128, nhead=4):
        super(MagNetAttention, self).__init__()
        self.embedding = nn.Linear(1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, x):
        # x shape: (batch, input_dim) -> needs to be (batch, input_dim, 1)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

def train_model(model, data, epochs=100):
    from .loss import MagnitudeLoss
    criterion = MagnitudeLoss(energy_weight=2.5)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    
    X = []
    y = []
    window = 10
    magnitudes = data['Magnitude'].values
    
    for i in range(len(magnitudes)-window):
        X.append(magnitudes[i:i+window])
        y.append(magnitudes[i+window])
        
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).view(-1, 1)

    print("🧠 [MagNet v2] Training the Brain with Attention Mechanism...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"   - Phase {epoch}: Energy Displacement Loss = {loss.item():.6f}")
    
    return model
