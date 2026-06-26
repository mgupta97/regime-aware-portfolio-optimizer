from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from metrics import performance_summary


PRICE_PATH = Path("data/processed/etf_prices.csv")
FIGURE_PATH = Path("figures/baseline_portfolio_comparison.png")


def load_prices(price_path: Path = PRICE_PATH) -> pd.DataFrame:
    prices = pd.read_csv(price_path, index_col="Date", parse_dates=True)

    if prices.empty:
        raise ValueError("Price file is empty.")

    return prices


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    return returns


def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Equal weight across all available ETFs.
    """
    num_assets = returns.shape[1]
    weights = pd.Series(1 / num_assets, index=returns.columns)
    portfolio_returns = returns @ weights
    return portfolio_returns


def sixty_forty_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Simple 60/40 portfolio:
    60% SPY
    40% IEF
    """
    required_assets = ["SPY", "IEF"]

    for asset in required_assets:
        if asset not in returns.columns:
            raise ValueError(f"{asset} is missing from the return data.")

    weights = pd.Series(
        {
            "SPY": 0.60,
            "IEF": 0.40,
        }
    )

    portfolio_returns = returns[weights.index] @ weights
    return portfolio_returns


def plot_portfolio_values(results: dict[str, pd.Series]) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for strategy_name, returns in results.items():
        portfolio_values = 10_000 * (1 + returns).cumprod()
        plt.plot(portfolio_values.index, portfolio_values, label=strategy_name)

    plt.title("Baseline Portfolio Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)

    print(f"Saved figure to: {FIGURE_PATH}")


def main() -> None:
    prices = load_prices()
    returns = calculate_daily_returns(prices)

    equal_weight_returns = equal_weight_portfolio(returns)
    sixty_forty_returns = sixty_forty_portfolio(returns)

    strategy_returns = {
        "Equal Weight": equal_weight_returns,
        "60/40 Portfolio": sixty_forty_returns,
    }

    summaries = [
        performance_summary(equal_weight_returns, "Equal Weight"),
        performance_summary(sixty_forty_returns, "60/40 Portfolio"),
    ]

    summary_df = pd.DataFrame(summaries)

    percent_columns = ["CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown"]

    print("\nBaseline Strategy Performance")
    print("=" * 80)

    display_df = summary_df.copy()
    display_df["Final Value"] = display_df["Final Value"].map(lambda x: f"${x:,.2f}")
    display_df["CAGR"] = display_df["CAGR"].map(lambda x: f"{x:.2%}")
    display_df["Volatility"] = display_df["Volatility"].map(lambda x: f"{x:.2%}")
    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].map(lambda x: f"{x:.2%}")

    print(display_df.to_string(index=False))

    plot_portfolio_values(strategy_returns)


if __name__ == "__main__":
    main()