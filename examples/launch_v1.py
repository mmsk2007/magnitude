import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import MagnitudeLoader
from src.magnet import MagNet, train_model
import torch
import time

def lana_insights(energy_level):
    """Lana AI Market Energy Interpretation"""
    print("\n🇸🇦 [Lana AI Insights]")
    if energy_level > 0.05:
        print("⚠️  Warning: Market energy is reaching critical levels. A massive displacement is imminent.")
    else:
        print("✅ Market energy is stable. Accumulation phase detected. Magnitude is building up.")
    print("--------------------------------------------------\n")

def main():
    print("🌌 MAGNITUDE v0.1.0 - The Physics of Price")
    print("Developed by Mohammad Alkhaldi & Lana AI\n")
    
    loader = MagnitudeLoader("BTC-USD")
    data = loader.fetch()
    
    model = MagNet(input_dim=10)
    train_model(model, data)
    
    # Predict next magnitude
    last_10 = torch.FloatTensor(data['Magnitude'].values[-10:])
    prediction = model(last_10.unsqueeze(0)).item()
    
    current_energy = data['Magnitude'].mean()
    
    print(f"\n📊 [Analysis Complete]")
    print(f"   - Current Market Energy: {current_energy:.4f}")
    print(f"   - Predicted Magnitude (Next Wave): {prediction:.4f}")
    
    lana_insights(prediction)
    
    print("🚀 Run complete. Magnitude captured.")

if __name__ == "__main__":
    main()
