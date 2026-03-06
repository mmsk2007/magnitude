import yfinance as yf
import pandas as pd
import numpy as np

class MagnitudeLoader:
    """Universal data fetcher for Magnitude framework."""
    def __init__(self, symbol="BTC-USD"):
        self.symbol = symbol

    def fetch(self, period="1y", interval="1d"):
        print(f"🌌 [Magnitude] Fetching energy data for {self.symbol}...")
        df = yf.download(self.symbol, period=period, interval=interval)
        return self._preprocess(df)

    def _preprocess(self, df):
        # Calculate Absolute Magnitude (The Core Philosophy)
        df['Returns'] = df['Close'].pct_change()
        df['Magnitude'] = df['Returns'].abs()
        
        # Volatility as proxy for Market Energy
        df['Energy'] = df['Magnitude'].rolling(window=10).mean()
        return df.dropna()
