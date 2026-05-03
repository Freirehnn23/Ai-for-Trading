from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

STRESS_DIR = BASE_DIR / "visualize" / "sequence" / "stress_test"
OUT_DIR = BASE_DIR / "visualize" / "sequence" / "comparison"

WEEK4_PATH = STRESS_DIR / "stress_test_week4_summary.csv"
WEEK5_PATH = STRESS_DIR / "stress_test_week5_summary.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not WEEK4_PATH.exists():
        raise FileNotFoundError(f"File Week4 tidak ditemukan: {WEEK4_PATH}")

    if not WEEK5_PATH.exists():
        raise FileNotFoundError(f"File Week5 tidak ditemukan: {WEEK5_PATH}")

    week4 = pd.read_csv(WEEK4_PATH)
    week5 = pd.read_csv(WEEK5_PATH)

    cols = [
        "scenario",
        "mean_return_pct",
        "mean_edge_vs_buy_hold_pct",
        "mean_max_drawdown_pct",
        "mean_trades",
        "mean_win_rate_pct",
        "mean_profit_factor",
    ]

    week4 = week4[cols].copy()
    week5 = week5[cols].copy()

    week4 = week4.add_prefix("week4_")
    week5 = week5.add_prefix("week5_")

    merged = pd.merge(
        week4,
        week5,
        left_on="week4_scenario",
        right_on="week5_scenario",
        how="inner",
    )

    result = pd.DataFrame()
    result["scenario"] = merged["week4_scenario"]

    result["week4_return_pct"] = merged["week4_mean_return_pct"]
    result["week5_return_pct"] = merged["week5_mean_return_pct"]
    result["diff_week5_minus_week4_return_pct"] = (
        merged["week5_mean_return_pct"] - merged["week4_mean_return_pct"]
    )

    result["week4_edge_vs_bh_pct"] = merged["week4_mean_edge_vs_buy_hold_pct"]
    result["week5_edge_vs_bh_pct"] = merged["week5_mean_edge_vs_buy_hold_pct"]
    result["diff_edge_pct"] = (
        merged["week5_mean_edge_vs_buy_hold_pct"]
        - merged["week4_mean_edge_vs_buy_hold_pct"]
    )

    result["week4_drawdown_pct"] = merged["week4_mean_max_drawdown_pct"]
    result["week5_drawdown_pct"] = merged["week5_mean_max_drawdown_pct"]

    result["week4_trades"] = merged["week4_mean_trades"]
    result["week5_trades"] = merged["week5_mean_trades"]

    result["week4_win_rate_pct"] = merged["week4_mean_win_rate_pct"]
    result["week5_win_rate_pct"] = merged["week5_mean_win_rate_pct"]

    result["week4_profit_factor"] = merged["week4_mean_profit_factor"]
    result["week5_profit_factor"] = merged["week5_mean_profit_factor"]

    out_path = OUT_DIR / "week5_vs_week4_stress_comparison.csv"
    result.to_csv(out_path, index=False)

    print("=" * 100)
    print("COMPARISON WEEK5 VS WEEK4")
    print("=" * 100)
    print(result.to_string(index=False))
    print("=" * 100)

    avg_diff_return = result["diff_week5_minus_week4_return_pct"].mean()

    if avg_diff_return > 0:
        print(f"RESULT: Week5 lebih baik rata-rata {avg_diff_return:+.2f}% dari Week4")
    else:
        print(f"RESULT: Week5 masih kalah rata-rata {avg_diff_return:+.2f}% dari Week4")

    print("=" * 100)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()