# 🌌 Magnitude: The Physics of Price Movement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-red.svg)]()

> **"Direction is noise. Magnitude is the only truth."**

Traditional trading models are obsessed with predicting *direction* (Long vs. Short). They fail because direction in efficient markets is a random walk dominated by noise. 

**Magnitude** is a paradigm shift. It is the world's first open-source trading framework that trains AI models specifically on the **Absolute Magnitude** of price changes, ignoring the sign. By focusing on the *force* of the move rather than its orientation, we capture the underlying energy of the market.

## 🧠 The Philosophy
In physics, velocity has direction, but energy (scalar) does not. Markets behave more like particles under stress than linear trends. Magnitude focuses on:
- **Volatility Surface Training:** Training on the magnitude of the next N-bars.
- **Energy Density:** Identifying zones where price displacement is inevitable.
- **Direction-Agnostic Arbitrage:** Profiting from the *existence* of movement, not the guess of its path.

## 🚀 Key Features
- **MagNet Core:** A transformer-based architecture optimized for scalar price displacement.
- **Universal Data Loader:** Seamlessly pull from Binance, yfinance, and CCXT.
- **Magnitude Loss Function:** A custom loss function that penalizes missing the *size* of the move more than the direction.

## 🛠 Tech Stack
- **Python 3.10+**
- **PyTorch** (Neural Network Core)
- **Pandas TA** (Feature Engineering)
- **Weights & Biases** (Experiment Tracking)

## 📈 Getting Started
```bash
git clone https://github.com/mmsk2007/magnitude.git
cd magnitude
pip install -r requirements.txt
python examples/train_magnitude_v1.py
```

## ⚖️ License
MIT. Built for the bold.

---
 
