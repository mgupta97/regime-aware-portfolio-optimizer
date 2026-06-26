from pathlib import Path
import pandas as pd
import yfinance as yf


TICKERS = [
    "SPY",  # U.S. large-cap equities
    "QQQ",  # U.S. tech/growth equities
    "IWM",  # U.S. small-cap equities
    "EFA",  # Developed international equities
    "EEM",  # Emerging markets equities
    "TLT",  # Long-term U.S. bonds
    "IEF",  # Intermediate-term U.S. bonds
    "GLD",  # Gold
    "VNQ",  # Real estate
    "DBC",  # Commodities
]

START_DATE = "2006-01-01"
OUTPUT_PATH = Path("data/processed/etf_prices.csv")


def fetch_adjusted_close(
    tickers: list[str] = TICKERS,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    """
    Download adjusted close prices using yfinance.

    auto_adjust=True means the Close price is adjusted for splits and dividends.
    """

    price_data = yf.download(
        tickers=tickers,
        start=start_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if price_data.empty:
        raise ValueError("No price data was downloaded. Check your internet connection or tickers.")

    if isinstance(price_data.columns, pd.MultiIndex):
        prices = price_data["Close"].copy()
    else:
        prices = price_data[["Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna(how="all")
    prices = prices.ffill()
    prices = prices.dropna()

    prices.index.name = "Date"

    return prices


def save_prices(prices: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path)
    print(f"Saved prices to: {output_path}")
    print(f"Rows: {len(prices)}")
    print(f"Columns: {list(prices.columns)}")


if __name__ == "__main__":
    prices = fetch_adjusted_close()
    save_prices(prices)
    print(prices.tail())