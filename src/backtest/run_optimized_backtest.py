from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest.engine import run_dynamic_rebalanced_backtest, run_rebalanced_backtest
from src.backtest.metrics import performance_summary
from src.models.optimizer import calculate_min_vol_weights, calculate_max_sharpe_weights



PRICE_PATH = Path("data/processed/etf_prices.csv")
FIGURE_PATH = Path("figures/optimized_portfolio_comparison.png")
WEIGHTS_PATH = Path("data/processed/optimized_weights.csv")


def load_prices(price_path: Path = PRICE_PATH) -> pd.DataFrame:
    prices = pd.read_csv(price_path, index_col="Date", parse_dates=True)

    if prices.empty:
        raise ValueError("Price file is empty.")

    return prices


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    return returns


def get_equal_weight_targets(assets: list[str]) -> pd.Series:
    num_assets = len(assets)
    return pd.Series(1 / num_assets, index=assets)


def get_sixty_forty_targets(assets: list[str]) -> pd.Series:
    weights = pd.Series(0.0, index=assets)

    if "SPY" not in assets:
        raise ValueError("SPY is required for the 60/40 portfolio.")

    if "IEF" not in assets:
        raise ValueError("IEF is required for the 60/40 portfolio.")

    weights["SPY"] = 0.60
    weights["IEF"] = 0.40

    return weights


def minimum_volatility_weight_function(
    historical_returns: pd.DataFrame,
    assets: list[str],
) -> pd.Series:
    return calculate_min_vol_weights(
        returns=historical_returns,
        assets=assets,
        max_weight=0.40,
    )

def maximum_sharpe_weight_function(
    historical_returns: pd.DataFrame,
    assets: list[str],
) -> pd.Series:
    return calculate_max_sharpe_weights(
        returns=historical_returns,
        assets=assets,
        max_weight=0.40,
        risk_free_rate=0.02,
    )

def plot_portfolio_values(results: list) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for result in results:
        plt.plot(
            result.portfolio_values.index,
            result.portfolio_values,
            label=result.strategy_name,
        )

    plt.title("Portfolio Strategy Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)

    print(f"Saved figure to: {FIGURE_PATH}")


def save_weight_history(results: list) -> None:
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    weight_frames = []

    for result in results:
        weights = result.weights.copy()
        weights["Strategy"] = result.strategy_name
        weights["Date"] = weights.index
        weight_frames.append(weights)

    all_weights = pd.concat(weight_frames)
    all_weights.to_csv(WEIGHTS_PATH, index=False)

    print(f"Saved weight history to: {WEIGHTS_PATH}")


def format_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    display_df = summary_df.copy()

    display_df["Final Value"] = display_df["Final Value"].map(lambda x: f"${x:,.2f}")
    display_df["CAGR"] = display_df["CAGR"].map(lambda x: f"{x:.2%}")
    display_df["Volatility"] = display_df["Volatility"].map(lambda x: f"{x:.2%}")
    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].map(lambda x: f"{x:.2%}")
    display_df["Average Rebalance Turnover"] = display_df["Average Rebalance Turnover"].map(
        lambda x: f"{x:.2%}"
    )
    display_df["Total Transaction Costs"] = display_df["Total Transaction Costs"].map(
        lambda x: f"${x:,.2f}"
    )

    return display_df


def print_latest_strategy_weights(result, title: str) -> None:
    latest_weights = result.weights.iloc[-1]
    latest_weights = latest_weights[latest_weights > 0.01].sort_values(ascending=False)

    print(f"\n{title}")
    print("=" * 80)

    display_weights = latest_weights.map(lambda x: f"{x:.2%}")
    print(display_weights.to_markdown())

def main() -> None:
    prices = load_prices()
    returns = calculate_daily_returns(prices)

    assets = list(returns.columns)

    equal_weight_targets = get_equal_weight_targets(assets)
    sixty_forty_targets = get_sixty_forty_targets(assets)

    transaction_cost_bps = 10

    equal_weight_result = run_rebalanced_backtest(
        returns=returns,
        target_weights=equal_weight_targets,
        strategy_name="Equal Weight Monthly Rebalanced",
        transaction_cost_bps=transaction_cost_bps,
    )

    sixty_forty_result = run_rebalanced_backtest(
        returns=returns,
        target_weights=sixty_forty_targets,
        strategy_name="60/40 Monthly Rebalanced",
        transaction_cost_bps=transaction_cost_bps,
    )

    min_vol_result = run_dynamic_rebalanced_backtest(
        returns=returns,
        weight_function=minimum_volatility_weight_function,
        strategy_name="Minimum Volatility Monthly Rebalanced",
        lookback_days=252,
        transaction_cost_bps=transaction_cost_bps,
        min_observations=60,
    )

    max_sharpe_result = run_dynamic_rebalanced_backtest(
        returns=returns,
        weight_function=maximum_sharpe_weight_function,
        strategy_name="Maximum Sharpe Ratio Monthly Rebalanced",
        lookback_days=252,
        transaction_cost_bps=transaction_cost_bps,
        min_observations=60,
    )

    results = [
        equal_weight_result,
        sixty_forty_result,
        min_vol_result,
        max_sharpe_result,
    ]

    summaries = []

    for result in results:
        summary = performance_summary(
            portfolio_returns=result.portfolio_returns,
            portfolio_name=result.strategy_name,
            portfolio_values=result.portfolio_values,
            turnover=result.turnover,
            transaction_costs=result.transaction_costs,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    print("\nOptimized Strategy Performance")
    print("=" * 100)
    print(f"Transaction cost assumption: {transaction_cost_bps} bps per dollar traded")
    print("-" * 100)

    display_df = format_summary_table(summary_df)
    print(display_df.to_markdown(index=False))

    print_latest_strategy_weights(min_vol_result, "Latest Minimum Volatility Weights")
    print_latest_strategy_weights(max_sharpe_result, "Latest Maximum Sharpe Weights")

    plot_portfolio_values(results)
    save_weight_history(results)


if __name__ == "__main__":
    main()