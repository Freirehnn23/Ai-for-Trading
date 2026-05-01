import pandas as pd
import numpy as np


def load_forex_data(file_name):
    df = pd.read_csv(file_name, encoding="utf-8-sig")

    # ── Rename kolom (dari header Indonesia) ────────────────────────
    df.columns = ["time", "close", "open", "high", "low", "volume", "change_pct"]

    # ── Parse tanggal format DD/MM/YYYY ─────────────────────────────
    df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y", dayfirst=True)

    # ── Konversi angka format Indonesia → float ──────────────────────
    # Contoh: "4.619,54" → 4619.54
    for col in ["close", "open", "high", "low"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('"', '', regex=False)
            .str.strip()
            .str.replace(".", "", regex=False)   # hapus pemisah ribuan
            .str.replace(",", ".", regex=False)  # ganti koma desimal → titik
            .astype(float)
        )

    # ── Tangani volume kosong ────────────────────────────────────────
    df["volume"] = (
        df["volume"]
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.strip()
        .replace("-", "0")
        .replace("", "0")
    )
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    # ── Sort dari terlama ke terbaru ─────────────────────────────────
    df = df.sort_values("time").reset_index(drop=True)

    # ── Tambah indikator ─────────────────────────────────────────────
    df["ma"]  = df["close"].rolling(10).mean()
    df["rsi"] = compute_rsi(df["close"], 14)

    # ── Drop NaN (dari rolling window) ──────────────────────────────
    df = df.dropna().reset_index(drop=True)

    print(f"[OK] Data loaded: {df.shape[0]} baris")
    print(f"     Periode: {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"     Close range: {df['close'].min():.2f} – {df['close'].max():.2f}")

    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

df = load_forex_data("data/Data_historis(23-26).csv")
print(df[["time", "close", "rsi"]].head())