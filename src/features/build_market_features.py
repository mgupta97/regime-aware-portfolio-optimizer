from pathlib import Path
import numpy as np
import pandas as pd


PRICE_PATH = Path("data/processed/etf_prices.csv")
FEATURES_OUTPUT_PATH = Path("data/processed/market_regime_features.csv")
TRADING_DAYS = 252


def load_prices(price_path: Path = PRICE_PATH) -> pd.DataFrame:
    prices = pd.read_csv(price_path, index_col="Date", parse_dates=True)

    if prices.empty:
        raise ValueError("Price file is empty.")

    return prices


def build_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly market-regime features using ETF prices.

    Features are based on:
    - equity momentum
    - bond momentum
    - commodity momentum
    - rolling volatility
    - stock/bond correlation
    - drawdown
    """

    required_assets = ["SPY", "QQQ", "IEF", "TLT", "GLD", "DBC"]

    missing_assets = [asset for asset in required_assets if asset not in prices.columns]

    if missing_assets:
        raise ValueError(f"Missing required assets: {missing_assets}")

    monthly_prices = prices.resample("ME").last()
    daily_returns = prices.pct_change()

    features = pd.DataFrame(index=monthly_prices.index)

    # Equity momentum
    features["SPY_3M_Return"] = monthly_prices["SPY"].pct_change(3)
    features["SPY_6M_Return"] = monthly_prices["SPY"].pct_change(6)
    features["SPY_12M_Return"] = monthly_prices["SPY"].pct_change(12)

    features["QQQ_3M_Return"] = monthly_prices["QQQ"].pct_change(3)

    # Bond and defensive asset momentum
    features["IEF_3M_Return"] = monthly_prices["IEF"].pct_change(3)
    features["TLT_3M_Return"] = monthly_prices["TLT"].pct_change(3)
    features["GLD_3M_Return"] = monthly_prices["GLD"].pct_change(3)

    # Commodity momentum
    features["DBC_3M_Return"] = monthly_prices["DBC"].pct_change(3)

    # Cross-asset spreads
    features["Equity_Bond_3M_Spread"] = (
        features["SPY_3M_Return"] - features["IEF_3M_Return"]
    )

    features["Commodity_Equity_3M_Spread"] = (
        features["DBC_3M_Return"] - features["SPY_3M_Return"]
    )

    # Rolling volatility
    features["SPY_3M_Volatility"] = (
        daily_returns["SPY"].rolling(63).std() * np.sqrt(TRADING_DAYS)
    ).resample("ME").last()

    features["QQQ_3M_Volatility"] = (
        daily_returns["QQQ"].rolling(63).std() * np.sqrt(TRADING_DAYS)
    ).resample("ME").last()

    features["Basket_3M_Volatility"] = (
        daily_returns.mean(axis=1).rolling(63).std() * np.sqrt(TRADING_DAYS)
    ).resample("ME").last()

    # Stock/bond correlation
    features["SPY_TLT_3M_Correlation"] = (
        daily_returns["SPY"].rolling(63).corr(daily_returns["TLT"])
    ).resample("ME").last()

    # Equity drawdown
    spy_running_max = monthly_prices["SPY"].cummax()
    features["SPY_Drawdown"] = monthly_prices["SPY"] / spy_running_max - 1

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()

    return features


def save_features(
    features: pd.DataFrame,
    output_path: Path = FEATURES_OUTPUT_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path)

    print(f"Saved market regime features to: {output_path}")
    print(f"Rows: {len(features)}")
    print(f"Columns: {list(features.columns)}")


if __name__ == "__main__":
    prices = load_prices()
    features = build_market_features(prices)
    save_features(features)

    print("\nLatest market regime features")
    print("=" * 80)
    print(features.tail().to_markdown())