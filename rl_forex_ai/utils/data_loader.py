import pandas as pd

def load_forex_data(file_name):
    df = pd.read_csv(file_name)

    # rename biar konsisten
    df.columns = [
        "time", "close", "open", "high", "low", "volume", "change_pct"
    ]

    # convert time
    df["time"] = pd.to_datetime(df["time"])

    # sort
    df = df.sort_values("time")

    # handle volume kosong
    df["volume"] = df["volume"].fillna(0)

    # === INDICATOR ===
    df["ma"] = df["close"].rolling(10).mean()
    df["rsi"] = compute_rsi(df["close"], 14)

    # drop NaN
    df = df.dropna().reset_index(drop=True)

    print("FINAL SHAPE:", df.shape)

    return df


def compute_rsi(series, period=14):
    delta = series.diff()

    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi