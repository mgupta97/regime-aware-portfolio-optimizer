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

SUMMARY_OUTPUT_PATH = Path("data/processed/true_regime_aware_strategy_summary.csv")
WEIGHTS_OUTPUT_PATH = Path("data/processed/true_regime_aware_weights.csv")
SIGNALS_OUTPUT_PATH = Path("data/processed/true_regime_aware_signals.csv")
RETURNS_OUTPUT_PATH = Path("data/processed/true_regime_aware_daily_returns.csv")

FIGURE_PATH = Path("figures/true_regime_aware_strategy_comparison.png")
WEIGHTS_FIGURE_PATH = Path("figures/true_regime_aware_weights.png")


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
    return prices.pct_change().dropna()


def get_equal_weight_targets(assets: list[str]) -> pd.Series:
    return pd.Series(1 / len(assets), index=assets)


def get_sixty_forty_targets(assets: list[str]) -> pd.Series:
    weights = pd.Series(0.0, index=assets)

    if "SPY" not in assets:
        raise ValueError("SPY is required for the 60/40 portfolio.")

    if "IEF" not in assets:
        raise ValueError("IEF is required for the 60/40 portfolio.")

    weights["SPY"] = 0.60
    weights["IEF"] = 0.40

    return weights


def validate_weights(weights: pd.Series, assets: list[str]) -> pd.Series:
    weights = weights.reindex(assets).fillna(0.0)

    if (weights < 0).any():
        raise ValueError("Negative weights are not allowed in this backtest.")

    weight_sum = weights.sum()

    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"Weights must sum to 1. Current sum: {weight_sum:.4f}")

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


def get_first_trading_day_each_month(returns: pd.DataFrame) -> set[pd.Timestamp]:
    rebalance_dates = returns.groupby(returns.index.to_period("M")).head(1).index
    return set(rebalance_dates)


def get_target_weights_for_strategy(
    selected_strategy: str,
    historical_returns: pd.DataFrame,
    assets: list[str],
    min_observations: int = 60,
) -> pd.Series:
    if selected_strategy == "Equal Weight":
        return get_equal_weight_targets(assets)

    if selected_strategy == "60/40":
        return get_sixty_forty_targets(assets)

    if len(historical_returns) < min_observations:
        return get_equal_weight_targets(assets)

    if selected_strategy == "Minimum Volatility":
        return minimum_volatility_weight_function(historical_returns, assets)

    if selected_strategy == "Maximum Sharpe":
        return maximum_sharpe_weight_function(historical_returns, assets)

    raise ValueError(f"Unknown strategy: {selected_strategy}")


def find_first_signal_date(
    returns: pd.DataFrame,
    regime_data: pd.DataFrame,
) -> pd.Timestamp:
    """
    First date where previous month's regime is available.
    """

    regime_months = set(regime_data.index.to_period("M"))

    for date in returns.index:
        previous_month = date.to_period("M") - 1

        if previous_month in regime_months:
            return date

    raise ValueError("No valid signal date found.")


def run_true_regime_aware_backtest(
    returns: pd.DataFrame,
    regime_data: pd.DataFrame,
    descriptions: dict[int, str],
    starting_value: float = 10_000,
    transaction_cost_bps: float = 10,
    lookback_days: int = 252,
    min_observations: int = 60,
) -> dict:
    """
    Runs a true regime-aware ETF portfolio.

    The portfolio:
    - checks the previous month's regime
    - selects the corresponding strategy
    - calculates actual target ETF weights
    - rebalances on the first trading day of each month
    - pays transaction costs on actual turnover
    - lets weights drift daily
    """

    if returns.empty:
        raise ValueError("Returns data is empty.")

    assets = list(returns.columns)
    transaction_cost_rate = transaction_cost_bps / 10_000

    regime_by_month = regime_data["Regime"].copy()
    regime_by_month.index = regime_by_month.index.to_period("M")

    rebalance_dates = get_first_trading_day_each_month(returns)
    start_date = find_first_signal_date(returns, regime_data)

    portfolio_value = starting_value
    current_weights = pd.Series(0.0, index=assets)

    portfolio_values = []
    portfolio_returns = []
    weight_history = []
    signal_history = []
    turnover_history = []
    transaction_cost_history = []

    active = False
    previous_portfolio_value = np.nan

    for row_number, (date, daily_returns) in enumerate(returns.iterrows()):
        if date < start_date:
            continue

        active = True

        turnover = 0.0
        transaction_cost = 0.0
        selected_strategy = None
        previous_month_regime = None
        regime_description = None

        if date in rebalance_dates or current_weights.sum() == 0:
            current_month = date.to_period("M")
            previous_month = current_month - 1

            if previous_month not in regime_by_month.index:
                continue

            previous_month_regime = int(regime_by_month.loc[previous_month])
            selected_strategy = REGIME_STRATEGY_MAP[previous_month_regime]
            regime_description = descriptions.get(previous_month_regime, "Unknown")

            lookback_start = max(0, row_number - lookback_days)
            historical_returns = returns.iloc[lookback_start:row_number]

            target_weights = get_target_weights_for_strategy(
                selected_strategy=selected_strategy,
                historical_returns=historical_returns,
                assets=assets,
                min_observations=min_observations,
            )

            target_weights = validate_weights(target_weights, assets)

            turnover = float((target_weights - current_weights).abs().sum())
            transaction_cost = portfolio_value * turnover * transaction_cost_rate

            portfolio_value -= transaction_cost
            current_weights = target_weights.copy()

        daily_portfolio_return = float(current_weights @ daily_returns)
        portfolio_value *= 1 + daily_portfolio_return

        if np.isnan(previous_portfolio_value):
            realized_return = np.nan
        else:
            realized_return = portfolio_value / previous_portfolio_value - 1

        previous_portfolio_value = portfolio_value

        updated_asset_values = current_weights * (1 + daily_returns)

        if updated_asset_values.sum() != 0:
            current_weights = updated_asset_values / updated_asset_values.sum()

        portfolio_values.append(portfolio_value)
        portfolio_returns.append(realized_return)
        weight_history.append(current_weights.copy())
        turnover_history.append(turnover)
        transaction_cost_history.append(transaction_cost)

        signal_history.append(
            {
                "Date": date,
                "Previous Month Regime": previous_month_regime,
                "Regime Description": regime_description,
                "Selected Strategy": selected_strategy,
                "Turnover": turnover,
                "Transaction Cost": transaction_cost,
            }
        )

    if not active:
        raise ValueError("Backtest never became active. Check regime dates.")

    portfolio_values = pd.Series(
        portfolio_values,
        index=returns.loc[start_date:].index[: len(portfolio_values)],
        name="True Regime-Aware Strategy",
    )

    portfolio_returns = pd.Series(
        portfolio_returns,
        index=portfolio_values.index,
        name="True Regime-Aware Strategy",
    ).dropna()

    weights = pd.DataFrame(
        weight_history,
        index=portfolio_values.index,
        columns=assets,
    )

    turnover = pd.Series(
        turnover_history,
        index=portfolio_values.index,
        name="Turnover",
    )

    transaction_costs = pd.Series(
        transaction_cost_history,
        index=portfolio_values.index,
        name="Transaction Costs",
    )

    signals = pd.DataFrame(signal_history)
    signals = signals.set_index("Date")

    return {
        "portfolio_values": portfolio_values,
        "portfolio_returns": portfolio_returns,
        "weights": weights,
        "turnover": turnover,
        "transaction_costs": transaction_costs,
        "signals": signals,
    }


def run_baseline_strategies(
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


def calculate_strategy_returns_from_results(results: list) -> pd.DataFrame:
    values = pd.concat(
        [result.portfolio_values.rename(result.strategy_name) for result in results],
        axis=1,
    )

    returns = values.pct_change().dropna(how="all")

    return returns


def calculate_portfolio_values(
    returns: pd.Series,
    starting_value: float = 10_000,
) -> pd.Series:
    return starting_value * (1 + returns).cumprod()


def build_comparison_returns(
    baseline_returns: pd.DataFrame,
    true_regime_aware_returns: pd.Series,
) -> pd.DataFrame:
    comparison_returns = baseline_returns.copy()
    comparison_returns["True Regime-Aware Strategy"] = true_regime_aware_returns

    comparison_returns = comparison_returns.loc[true_regime_aware_returns.index]
    comparison_returns = comparison_returns.dropna(how="any")

    return comparison_returns


def summarize_comparison(
    comparison_returns: pd.DataFrame,
    true_regime_aware_result: dict,
) -> pd.DataFrame:
    summaries = []

    for strategy in comparison_returns.columns:
        strategy_returns = comparison_returns[strategy].dropna()
        strategy_values = calculate_portfolio_values(strategy_returns)

        if strategy == "True Regime-Aware Strategy":
            summary = performance_summary(
                portfolio_returns=strategy_returns,
                portfolio_name=strategy,
                portfolio_values=strategy_values,
                turnover=true_regime_aware_result["turnover"].loc[strategy_returns.index],
                transaction_costs=true_regime_aware_result["transaction_costs"].loc[
                    strategy_returns.index
                ],
            )
        else:
            summary = performance_summary(
                portfolio_returns=strategy_returns,
                portfolio_name=strategy,
                portfolio_values=strategy_values,
            )

        summaries.append(summary)

    return pd.DataFrame(summaries)


def format_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    display_df = summary_df.copy()

    display_df["Final Value"] = display_df["Final Value"].map(lambda x: f"${x:,.2f}")
    display_df["CAGR"] = display_df["CAGR"].map(lambda x: f"{x:.2%}")
    display_df["Volatility"] = display_df["Volatility"].map(lambda x: f"{x:.2%}")
    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].map(lambda x: f"{x:.2%}")

    if "Average Rebalance Turnover" in display_df.columns:
        display_df["Average Rebalance Turnover"] = display_df[
            "Average Rebalance Turnover"
        ].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")

    if "Total Transaction Costs" in display_df.columns:
        display_df["Total Transaction Costs"] = display_df[
            "Total Transaction Costs"
        ].map(lambda x: "" if pd.isna(x) else f"${x:,.2f}")

    return display_df


def plot_strategy_comparison(comparison_returns: pd.DataFrame) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for strategy in comparison_returns.columns:
        portfolio_values = calculate_portfolio_values(comparison_returns[strategy])
        plt.plot(portfolio_values.index, portfolio_values, label=strategy)

    plt.title("True Regime-Aware ETF Portfolio vs Baselines")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)

    print(f"Saved figure to: {FIGURE_PATH}")


def plot_regime_aware_weights(weights: pd.DataFrame) -> None:
    WEIGHTS_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    monthly_weights = weights.resample("ME").last()

    ax = monthly_weights.plot.area(figsize=(12, 6))

    ax.set_title("True Regime-Aware Strategy ETF Weights")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Weight")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(WEIGHTS_FIGURE_PATH, dpi=300)
    plt.close()

    print(f"Saved weights figure to: {WEIGHTS_FIGURE_PATH}")


def print_strategy_rule(descriptions: dict[int, str]) -> None:
    print("\nTrue Regime-Aware Strategy Rule")
    print("=" * 100)

    for regime, strategy in REGIME_STRATEGY_MAP.items():
        description = descriptions.get(regime, "Unknown")
        print(f"Previous month Regime {regime} ({description}) -> Use {strategy}")


def print_strategy_usage(signals: pd.DataFrame) -> None:
    print("\nSelected Strategy Usage on Rebalance Days")
    print("=" * 100)

    rebalance_signals = signals.dropna(subset=["Selected Strategy"])

    usage = rebalance_signals["Selected Strategy"].value_counts(normalize=True).sort_index()
    usage = usage.map(lambda x: f"{x:.2%}")

    print(usage.to_markdown())


def print_latest_weights(weights: pd.DataFrame) -> None:
    latest_weights = weights.iloc[-1]
    latest_weights = latest_weights[latest_weights > 0.01].sort_values(ascending=False)

    print("\nLatest True Regime-Aware ETF Weights")
    print("=" * 100)

    display_weights = latest_weights.map(lambda x: f"{x:.2%}")
    print(display_weights.to_markdown())


def print_latest_signal(signals: pd.DataFrame) -> None:
    latest_signal = signals.dropna(subset=["Selected Strategy"]).iloc[-1]

    print("\nLatest True Regime-Aware Signal")
    print("=" * 100)
    print(f"Date: {latest_signal.name.date()}")
    print(f"Previous Month Regime: {latest_signal['Previous Month Regime']}")
    print(f"Regime Description: {latest_signal['Regime Description']}")
    print(f"Selected Strategy: {latest_signal['Selected Strategy']}")
    print(f"Turnover: {latest_signal['Turnover']:.2%}")
    print(f"Transaction Cost: ${latest_signal['Transaction Cost']:,.2f}")


def save_outputs(
    summary_df: pd.DataFrame,
    true_result: dict,
    comparison_returns: pd.DataFrame,
) -> None:
    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    true_result["weights"].to_csv(WEIGHTS_OUTPUT_PATH)
    true_result["signals"].to_csv(SIGNALS_OUTPUT_PATH)
    comparison_returns.to_csv(RETURNS_OUTPUT_PATH)

    print(f"Saved summary to: {SUMMARY_OUTPUT_PATH}")
    print(f"Saved weights to: {WEIGHTS_OUTPUT_PATH}")
    print(f"Saved signals to: {SIGNALS_OUTPUT_PATH}")
    print(f"Saved comparison returns to: {RETURNS_OUTPUT_PATH}")


def main() -> None:
    prices = load_prices()
    returns = calculate_daily_returns(prices)

    transaction_cost_bps = 10

    regime_data, descriptions = build_regime_data(prices)

    baseline_results = run_baseline_strategies(
        returns=returns,
        transaction_cost_bps=transaction_cost_bps,
    )

    baseline_returns = calculate_strategy_returns_from_results(baseline_results)

    true_result = run_true_regime_aware_backtest(
        returns=returns,
        regime_data=regime_data,
        descriptions=descriptions,
        starting_value=10_000,
        transaction_cost_bps=transaction_cost_bps,
        lookback_days=252,
        min_observations=60,
    )

    comparison_returns = build_comparison_returns(
        baseline_returns=baseline_returns,
        true_regime_aware_returns=true_result["portfolio_returns"],
    )

    summary_df = summarize_comparison(
        comparison_returns=comparison_returns,
        true_regime_aware_result=true_result,
    )

    print("\nTrue Regime-Aware ETF Portfolio Backtest")
    print("=" * 100)
    print(f"Transaction cost assumption: {transaction_cost_bps} bps per dollar traded")
    print("Signal logic: previous month's detected regime selects this month's ETF weights")
    print("Rebalance logic: first trading day of each month")
    print("-" * 100)

    print_strategy_rule(descriptions)

    print("\nStrategy Performance Comparison")
    print("=" * 100)

    display_df = format_summary_table(summary_df)
    print(display_df.to_markdown(index=False))

    print_strategy_usage(true_result["signals"])
    print_latest_signal(true_result["signals"])
    print_latest_weights(true_result["weights"])

    plot_strategy_comparison(comparison_returns)
    plot_regime_aware_weights(true_result["weights"])

    save_outputs(
        summary_df=summary_df,
        true_result=true_result,
        comparison_returns=comparison_returns,
    )


if __name__ == "__main__":
    main()