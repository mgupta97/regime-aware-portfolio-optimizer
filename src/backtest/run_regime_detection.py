from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.features.build_market_features import build_market_features
from src.models.regime_model import (
    assign_regime_descriptions,
    fit_kmeans_regimes,
    summarize_regimes,
)


PRICE_PATH = Path("data/processed/etf_prices.csv")
FEATURES_PATH = Path("data/processed/market_regime_features.csv")
REGIME_OUTPUT_PATH = Path("data/processed/market_regime_labels.csv")

REGIME_TIMELINE_FIGURE_PATH = Path("figures/market_regime_timeline.png")
SPY_REGIME_FIGURE_PATH = Path("figures/spy_with_market_regimes.png")


def load_prices(price_path: Path = PRICE_PATH) -> pd.DataFrame:
    prices = pd.read_csv(price_path, index_col="Date", parse_dates=True)

    if prices.empty:
        raise ValueError("Price file is empty.")

    return prices


def save_regime_data(regime_data: pd.DataFrame) -> None:
    REGIME_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    regime_data.to_csv(REGIME_OUTPUT_PATH)

    print(f"Saved regime labels to: {REGIME_OUTPUT_PATH}")


def format_regime_summary(summary: pd.DataFrame) -> pd.DataFrame:
    display_df = summary.copy()

    percent_columns = [
        "SPY_3M_Return",
        "SPY_6M_Return",
        "SPY_12M_Return",
        "QQQ_3M_Return",
        "IEF_3M_Return",
        "TLT_3M_Return",
        "GLD_3M_Return",
        "DBC_3M_Return",
        "Equity_Bond_3M_Spread",
        "Commodity_Equity_3M_Spread",
        "SPY_3M_Volatility",
        "QQQ_3M_Volatility",
        "Basket_3M_Volatility",
        "SPY_Drawdown",
    ]

    for column in percent_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda x: f"{x:.2%}")

    if "SPY_TLT_3M_Correlation" in display_df.columns:
        display_df["SPY_TLT_3M_Correlation"] = display_df[
            "SPY_TLT_3M_Correlation"
        ].map(lambda x: f"{x:.2f}")

    if "Observations" in display_df.columns:
        display_df["Observations"] = display_df["Observations"].astype(int)

    return display_df


def plot_regime_timeline(regime_data: pd.DataFrame) -> None:
    REGIME_TIMELINE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.scatter(
        regime_data.index,
        regime_data["Regime"],
        c=regime_data["Regime"],
        s=25,
    )

    plt.title("Detected Market Regimes Over Time")
    plt.xlabel("Date")
    plt.ylabel("Regime")
    plt.yticks(sorted(regime_data["Regime"].unique()))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REGIME_TIMELINE_FIGURE_PATH, dpi=300)

    print(f"Saved regime timeline figure to: {REGIME_TIMELINE_FIGURE_PATH}")


def plot_spy_with_regimes(
    prices: pd.DataFrame,
    regime_data: pd.DataFrame,
) -> None:
    SPY_REGIME_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    monthly_spy = prices["SPY"].resample("ME").last()
    monthly_spy = monthly_spy.reindex(regime_data.index).dropna()

    aligned_regimes = regime_data.loc[monthly_spy.index, "Regime"]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(monthly_spy.index, monthly_spy.values, label="SPY")
    ax1.set_title("SPY Price with Detected Market Regimes")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("SPY Price")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.scatter(
        monthly_spy.index,
        aligned_regimes.values,
        c=aligned_regimes.values,
        s=20,
        alpha=0.7,
    )
    ax2.set_ylabel("Regime")
    ax2.set_yticks(sorted(regime_data["Regime"].unique()))

    fig.tight_layout()
    fig.savefig(SPY_REGIME_FIGURE_PATH, dpi=300)

    print(f"Saved SPY regime figure to: {SPY_REGIME_FIGURE_PATH}")


def print_latest_regime(
    regime_data: pd.DataFrame,
    descriptions: dict[int, str],
) -> None:
    latest_date = regime_data.index[-1]
    latest_regime = int(regime_data["Regime"].iloc[-1])
    latest_description = descriptions.get(latest_regime, "Unknown")

    print("\nLatest Detected Market Regime")
    print("=" * 80)
    print(f"Date: {latest_date.date()}")
    print(f"Regime: {latest_regime}")
    print(f"Description: {latest_description}")


def main() -> None:
    prices = load_prices()

    features = build_market_features(prices)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH)

    regime_data = fit_kmeans_regimes(
        features=features,
        n_regimes=4,
        random_state=42,
    )

    save_regime_data(regime_data)

    summary = summarize_regimes(regime_data)
    descriptions = assign_regime_descriptions(summary)

    print("\nMarket Regime Summary")
    print("=" * 100)

    display_summary = format_regime_summary(summary)
    print(display_summary.to_markdown())

    print("\nSuggested Regime Descriptions")
    print("=" * 100)

    for regime, description in descriptions.items():
        print(f"Regime {regime}: {description}")

    print_latest_regime(regime_data, descriptions)

    plot_regime_timeline(regime_data)
    plot_spy_with_regimes(prices, regime_data)


if __name__ == "__main__":
    main()