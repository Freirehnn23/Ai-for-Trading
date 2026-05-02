import pandas as pd


def _to_float_id(value):
    """
    Konversi angka format Indonesia / market data ke float.

    Contoh:
        "4.619,54" -> 4619.54
        "121.32K"  -> 121320.0
        "1.5M"     -> 1500000.0
        "1,82%"    -> 1.82
        "-"         -> 0.0
        ""          -> 0.0
    """
    if pd.isna(value):
        return 0.0

    value = str(value).strip()
    value = value.replace('"', "")
    value = value.replace("%", "")
    value = value.replace(" ", "")

    if value == "" or value == "-":
        return 0.0

    multiplier = 1.0

    suffix = value[-1].upper()

    if suffix == "K":
        multiplier = 1_000.0
        value = value[:-1]
    elif suffix == "M":
        multiplier = 1_000_000.0
        value = value[:-1]
    elif suffix == "B":
        multiplier = 1_000_000_000.0
        value = value[:-1]

    # Format Indonesia: 4.619,54 -> 4619.54
    if "." in value and "," in value:
        value = value.replace(".", "")
        value = value.replace(",", ".")

    # Format koma desimal: 1,82 -> 1.82
    elif "," in value:
        value = value.replace(",", ".")

    # Format titik:
    # 121.32K -> 121.32
    # 4.619   -> 4619 jika dianggap ribuan
    elif "." in value:
        parts = value.split(".")
        if len(parts) == 2 and len(parts[1]) == 3 and multiplier == 1.0:
            value = value.replace(".", "")

    try:
        return float(value) * multiplier
    except ValueError:
        return 0.0


def compute_rsi(series, period=14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def load_forex_data(file_name):
    """
    Load data historis forex / index dari CSV.

    Format CSV asli:
        Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol., Perubahan%

    Output:
        time, close, open, high, low, volume, change_pct, ma, rsi
    """

    df = pd.read_csv(file_name, encoding="utf-8-sig")

    # Bersihkan nama kolom
    df.columns = [str(col).strip().replace('"', "") for col in df.columns]

    rename_map = {
        "Tanggal": "time",
        "Terakhir": "close",
        "Pembukaan": "open",
        "Tertinggi": "high",
        "Terendah": "low",
        "Vol.": "volume",
        "Perubahan%": "change_pct",
    }

    df = df.rename(columns=rename_map)

    required_columns = ["time", "open", "high", "low", "close"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing_columns}. "
            f"Kolom terbaca: {list(df.columns)}"
        )

    # Parse tanggal
    df["time"] = pd.to_datetime(
        df["time"],
        format="%d/%m/%Y",
        dayfirst=True,
        errors="coerce",
    )

    # Konversi OHLC
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].apply(_to_float_id)

    # Konversi volume
    if "volume" in df.columns:
        df["volume"] = df["volume"].apply(_to_float_id)
    else:
        df["volume"] = 0.0

    # Konversi perubahan persen
    if "change_pct" in df.columns:
        df["change_pct"] = df["change_pct"].apply(_to_float_id)
    else:
        df["change_pct"] = 0.0

    # Drop tanggal invalid
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    # Sort dari data lama ke baru
    df = df.sort_values("time").reset_index(drop=True)

    # Indikator sederhana
    df["ma"] = df["close"].rolling(10).mean()
    df["rsi"] = compute_rsi(df["close"], 14)

    # Drop NaN dari rolling indicator
    df = df.dropna().reset_index(drop=True)

    print(f"[OK] Data loaded: {df.shape[0]} baris")
    print(f"     Periode: {df['time'].min().date()} -> {df['time'].max().date()}")
    print(f"     Close range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"     Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    df = load_forex_data("data/Data_historis(23-26).csv")
    print(df[["time", "open", "high", "low", "close", "volume", "rsi"]].head())