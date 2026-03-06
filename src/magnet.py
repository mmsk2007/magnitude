import torch
import torch.nn as nn
import torch.optim as optim

class MagNet(nn.Module):
    """
    MagNet: A scalar-focused architecture that predicts price displacement force.
    """
    def __init__(self, input_dim=10):
        super(MagNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Magnitude must always be positive
        )

    def forward(self, x):
        return self.network(x)

def train_model(model, data, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare windowed data
    X = []
    y = []
    window = 10
    magnitudes = data['Magnitude'].values
    
    for i in range(len(magnitudes)-window):
        X.append(magnitudes[i:i+window])
        y.append(magnitudes[i+window])
        
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).view(-1, 1)

    print("🧠 [MagNet] Initiating training on price magnitude energy...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"   - Phase {epoch}: Energy Loss = {loss.item():.6f}")
    
    return model
