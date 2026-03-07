import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import MagnitudeLoader
from src.magnet import MagNetAttention, train_model
import torch

def main():
    print("🌌 MAGNITUDE v0.2.0 - The Attention Era")
    print("Focusing on Price Displacement Energy\n")
    
    loader = MagnitudeLoader("BTC-USD")
    data = loader.fetch()
    
    # Initialize with Attention-based Brain
    model = MagNetAttention(input_dim=10)
    train_model(model, data, epochs=100)
    
    # Capture the latest energy state
    last_10 = torch.FloatTensor(data['Magnitude'].values[-10:])
    prediction = model(last_10.unsqueeze(0)).item()
    
    current_avg = data['Magnitude'].mean()
    
    print(f"\n📊 [Advanced Analysis]")
    print(f"   - Historical Energy Mean: {current_avg:.4f}")
    print(f"   - Predicted Magnitude Burst: {prediction:.4f}")
    
    if prediction > current_avg * 1.5:
        print("\n🔥 ALERT: High probability of a massive displacement wave.")
    else:
        print("\n✨ MARKET: Normal energy levels maintained.")
    
    print("\n🚀 Magnitude captured successfully.")

if __name__ == "__main__":
    main()
