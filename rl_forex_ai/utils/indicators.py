import pandas as pd

def add_indicators(df):
    # contoh indikator sederhana

    # Moving Average
    df["ma"] = df["close"].rolling(10).mean()

    # RSI sederhana
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df