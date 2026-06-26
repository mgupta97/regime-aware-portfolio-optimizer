from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest.engine import run_dynamic_rebalanced_backtest, run_rebalanced_backtest
from src.backtest.metrics import performance_summary
from src.features.build_market_features import build_market_features
from src.models.optimizer import calculate_min_vol_weights, calculate_max_sharpe_weights
from src.models.regime_model import (
    assign_regime_descriptions,
    fit_kmeans_regimes,
    summarize_regimes,
)


PRICE_PATH = Path("data/processed/etf_prices.csv")

SIGNALS_OUTPUT_PATH = Path("data/processed/regime_aware_signals.csv")
RETURNS_OUTPUT_PATH = Path("data/processed/regime_aware_daily_returns.csv")
SUMMARY_OUTPUT_PATH = Path("data/processed/regime_aware_strategy_summary.csv")

FIGURE_PATH = Path("figures/regime_aware_strategy_comparison.png")


REGIME_STRATEGY_MAP = {
    0: "60/40",
    1: "Maximum Sharpe",
    2: "Minimum Volatility",
    3: "Equal Weight",
}


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


def run_all_strategies(
    returns: pd.DataFrame,
    transaction_cost_bps: float = 10,
) -> list:
    assets = list(returns.columns)

    equal_weight_result = run_rebalanced_backtest(
        returns=returns,
        target_weights=get_equal_weight_targets(assets),
        strategy_name="Equal Weight",
        transaction_cost_bps=transaction_cost_bps,
    )

    sixty_forty_result = run_rebalanced_backtest(
        returns=returns,
        target_weights=get_sixty_forty_targets(assets),
        strategy_name="60/40",
        transaction_cost_bps=transaction_cost_bps,
    )

    min_vol_result = run_dynamic_rebalanced_backtest(
        returns=returns,
        weight_function=minimum_volatility_weight_function,
        strategy_name="Minimum Volatility",
        lookback_days=252,
        transaction_cost_bps=transaction_cost_bps,
        min_observations=60,
    )

    max_sharpe_result = run_dynamic_rebalanced_backtest(
        returns=returns,
        weight_function=maximum_sharpe_weight_function,
        strategy_name="Maximum Sharpe",
        lookback_days=252,
        transaction_cost_bps=transaction_cost_bps,
        min_observations=60,
    )

    return [
        equal_weight_result,
        sixty_forty_result,
        min_vol_result,
        max_sharpe_result,
    ]


def build_regime_data(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    features = build_market_features(prices)

    regime_data = fit_kmeans_regimes(
        features=features,
        n_regimes=4,
        random_state=42,
    )

    summary = summarize_regimes(regime_data)
    descriptions = assign_regime_descriptions(summary)

    return regime_data, descriptions


def calculate_strategy_daily_returns(results: list) -> pd.DataFrame:
    strategy_values = pd.concat(
        [result.portfolio_values.rename(result.strategy_name) for result in results],
        axis=1,
    )

    strategy_returns = strategy_values.pct_change().dropna(how="all")

    return strategy_returns


def build_regime_aware_signals(
    strategy_returns: pd.DataFrame,
    regime_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build daily strategy-selection signals using previous month's regime.

    Example:
    - If January was Regime 0, use the Regime 0 strategy during February.
    """

    regime_by_month = regime_data["Regime"].copy()
    regime_by_month.index = regime_by_month.index.to_period("M")

    signals = pd.DataFrame(index=strategy_returns.index)
    signals["Month"] = signals.index.to_period("M")
    signals["Previous Month"] = signals["Month"].apply(lambda period: period - 1)

    signals["Previous Month Regime"] = signals["Previous Month"].map(regime_by_month)
    signals = signals.dropna(subset=["Previous Month Regime"])

    signals["Previous Month Regime"] = signals["Previous Month Regime"].astype(int)
    signals["Selected Strategy"] = signals["Previous Month Regime"].map(
        REGIME_STRATEGY_MAP
    )

    signals = signals.dropna(subset=["Selected Strategy"])

    return signals


def build_regime_aware_returns(
    strategy_returns: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.Series:
    """
    Create daily regime-aware returns by selecting the return of the strategy
    chosen by the previous month's regime.
    """

    aligned_returns = strategy_returns.loc[signals.index]

    regime_aware_returns = []

    for date in aligned_returns.index:
        selected_strategy = signals.loc[date, "Selected Strategy"]
        daily_return = aligned_returns.loc[date, selected_strategy]
        regime_aware_returns.append(daily_return)

    regime_aware_returns = pd.Series(
        regime_aware_returns,
        index=aligned_returns.index,
        name="Regime-Aware Strategy",
    )

    return regime_aware_returns


def build_comparison_returns(
    strategy_returns: pd.DataFrame,
    regime_aware_returns: pd.Series,
) -> pd.DataFrame:
    comparison_returns = strategy_returns.copy()
    comparison_returns["Regime-Aware Strategy"] = regime_aware_returns

    comparison_returns = comparison_returns.loc[regime_aware_returns.index]
    comparison_returns = comparison_returns.dropna(how="any")

    return comparison_returns


def calculate_portfolio_values(
    returns: pd.Series,
    starting_value: float = 10_000,
) -> pd.Series:
    return starting_value * (1 + returns).cumprod()


def summarize_comparison(comparison_returns: pd.DataFrame) -> pd.DataFrame:
    summaries = []

    for strategy in comparison_returns.columns:
        strategy_returns = comparison_returns[strategy].dropna()
        strategy_values = calculate_portfolio_values(strategy_returns)

        summary = performance_summary(
            portfolio_returns=strategy_returns,
            portfolio_name=strategy,
            portfolio_values=strategy_values,
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    return summary_df


def format_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    display_df = summary_df.copy()

    display_df["Final Value"] = display_df["Final Value"].map(lambda x: f"${x:,.2f}")
    display_df["CAGR"] = display_df["CAGR"].map(lambda x: f"{x:.2%}")
    display_df["Volatility"] = display_df["Volatility"].map(lambda x: f"{x:.2%}")
    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].map(lambda x: f"{x:.2%}")

    return display_df


def plot_strategy_comparison(comparison_returns: pd.DataFrame) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for strategy in comparison_returns.columns:
        portfolio_values = calculate_portfolio_values(comparison_returns[strategy])
        plt.plot(portfolio_values.index, portfolio_values, label=strategy)

    plt.title("Regime-Aware Strategy vs Baselines")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)

    print(f"Saved figure to: {FIGURE_PATH}")


def print_strategy_rule(descriptions: dict[int, str]) -> None:
    print("\nRegime-Aware Strategy Rule")
    print("=" * 100)

    for regime, strategy in REGIME_STRATEGY_MAP.items():
        description = descriptions.get(regime, "Unknown")
        print(f"Previous month Regime {regime} ({description}) -> Use {strategy}")


def print_strategy_usage(signals: pd.DataFrame) -> None:
    print("\nSelected Strategy Usage")
    print("=" * 100)

    usage = signals["Selected Strategy"].value_counts(normalize=True).sort_index()
    usage = usage.map(lambda x: f"{x:.2%}")

    print(usage.to_markdown())


def print_latest_signal(signals: pd.DataFrame) -> None:
    latest_date = signals.index[-1]
    latest_signal = signals.iloc[-1]

    print("\nLatest Regime-Aware Signal")
    print("=" * 100)
    print(f"Date: {latest_date.date()}")
    print(f"Previous Month Regime: {latest_signal['Previous Month Regime']}")
    print(f"Selected Strategy: {latest_signal['Selected Strategy']}")


def save_outputs(
    signals: pd.DataFrame,
    comparison_returns: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    SIGNALS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    signals.to_csv(SIGNALS_OUTPUT_PATH)
    comparison_returns.to_csv(RETURNS_OUTPUT_PATH)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print(f"Saved signals to: {SIGNALS_OUTPUT_PATH}")
    print(f"Saved daily returns to: {RETURNS_OUTPUT_PATH}")
    print(f"Saved summary to: {SUMMARY_OUTPUT_PATH}")


def main() -> None:
    prices = load_prices()
    returns = calculate_daily_returns(prices)

    transaction_cost_bps = 10

    results = run_all_strategies(
        returns=returns,
        transaction_cost_bps=transaction_cost_bps,
    )

    regime_data, descriptions = build_regime_data(prices)

    strategy_returns = calculate_strategy_daily_returns(results)

    signals = build_regime_aware_signals(
        strategy_returns=strategy_returns,
        regime_data=regime_data,
    )

    regime_aware_returns = build_regime_aware_returns(
        strategy_returns=strategy_returns,
        signals=signals,
    )

    comparison_returns = build_comparison_returns(
        strategy_returns=strategy_returns,
        regime_aware_returns=regime_aware_returns,
    )

    summary_df = summarize_comparison(comparison_returns)

    print("\nRegime-Aware Strategy Backtest")
    print("=" * 100)
    print(f"Transaction cost assumption inside component strategies: {transaction_cost_bps} bps")
    print("Signal logic: previous month's detected regime selects this month's strategy")
    print("-" * 100)

    print_strategy_rule(descriptions)

    print("\nStrategy Performance Comparison")
    print("=" * 100)
    display_df = format_summary_table(summary_df)
    print(display_df.to_markdown(index=False))

    print_strategy_usage(signals)
    print_latest_signal(signals)

    plot_strategy_comparison(comparison_returns)

    save_outputs(
        signals=signals,
        comparison_returns=comparison_returns,
        summary_df=summary_df,
    )


if __name__ == "__main__":
    main()