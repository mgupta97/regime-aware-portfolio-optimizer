from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest.engine import run_dynamic_rebalanced_backtest, run_rebalanced_backtest
from src.features.build_market_features import build_market_features
from src.models.optimizer import calculate_min_vol_weights, calculate_max_sharpe_weights
from src.models.regime_model import (
    assign_regime_descriptions,
    fit_kmeans_regimes,
    summarize_regimes,
)


PRICE_PATH = Path("data/processed/etf_prices.csv")

REGIME_OUTPUT_PATH = Path("data/processed/market_regime_labels.csv")
REGIME_STRATEGY_OUTPUT_PATH = Path("data/processed/strategy_performance_by_regime.csv")

RETURN_FIGURE_PATH = Path("figures/regime_strategy_annualized_return.png")
SHARPE_FIGURE_PATH = Path("figures/regime_strategy_sharpe.png")
CUMULATIVE_FIGURE_PATH = Path("figures/regime_strategy_cumulative_return.png")


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


def run_all_strategies(returns: pd.DataFrame, transaction_cost_bps: float = 10) -> list:
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

    REGIME_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    regime_data.to_csv(REGIME_OUTPUT_PATH)

    return regime_data, descriptions


def calculate_monthly_strategy_returns(results: list) -> pd.DataFrame:
    portfolio_values = pd.concat(
        [result.portfolio_values.rename(result.strategy_name) for result in results],
        axis=1,
    )

    monthly_values = portfolio_values.resample("ME").last()
    monthly_returns = monthly_values.pct_change().dropna(how="all")

    monthly_returns.index.name = "Date"

    return monthly_returns


def align_returns_with_regimes(
    monthly_returns: pd.DataFrame,
    regime_data: pd.DataFrame,
) -> pd.DataFrame:
    regime_series = regime_data["Regime"].copy()
    regime_series.index = regime_series.index.to_period("M")

    merged = monthly_returns.copy()
    merged["Month"] = merged.index.to_period("M")
    merged["Regime"] = merged["Month"].map(regime_series)
    merged = merged.drop(columns=["Month"])
    merged = merged.dropna(subset=["Regime"])
    merged["Regime"] = merged["Regime"].astype(int)

    return merged


def summarize_strategy_performance_by_regime(
    monthly_returns_with_regimes: pd.DataFrame,
    descriptions: dict[int, str],
) -> pd.DataFrame:
    strategy_columns = [
        col for col in monthly_returns_with_regimes.columns if col != "Regime"
    ]

    records = []

    for regime in sorted(monthly_returns_with_regimes["Regime"].unique()):
        regime_subset = monthly_returns_with_regimes[
            monthly_returns_with_regimes["Regime"] == regime
        ]

        for strategy in strategy_columns:
            strategy_returns = regime_subset[strategy].dropna()

            if strategy_returns.empty:
                continue

            mean_monthly_return = strategy_returns.mean()
            monthly_volatility = strategy_returns.std()
            cumulative_return = (1 + strategy_returns).prod() - 1

            annualized_return = (1 + mean_monthly_return) ** 12 - 1
            annualized_volatility = monthly_volatility * np.sqrt(12)

            if annualized_volatility == 0 or np.isnan(annualized_volatility):
                sharpe_ratio = np.nan
            else:
                sharpe_ratio = annualized_return / annualized_volatility

            hit_rate = (strategy_returns > 0).mean()
            best_month = strategy_returns.max()
            worst_month = strategy_returns.min()

            records.append(
                {
                    "Regime": regime,
                    "Regime Description": descriptions.get(regime, "Unknown"),
                    "Strategy": strategy,
                    "Months": len(strategy_returns),
                    "Mean Monthly Return": mean_monthly_return,
                    "Annualized Return": annualized_return,
                    "Annualized Volatility": annualized_volatility,
                    "Sharpe Ratio": sharpe_ratio,
                    "Cumulative Return In Regime": cumulative_return,
                    "Hit Rate": hit_rate,
                    "Best Month": best_month,
                    "Worst Month": worst_month,
                }
            )

    summary = pd.DataFrame(records)

    return summary


def add_best_strategy_flags(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()

    summary["Best By Sharpe"] = False
    summary["Best By Annualized Return"] = False
    summary["Best By Cumulative Return"] = False

    for regime in summary["Regime"].unique():
        regime_mask = summary["Regime"] == regime
        regime_data = summary.loc[regime_mask]

        best_sharpe_index = regime_data["Sharpe Ratio"].idxmax()
        best_return_index = regime_data["Annualized Return"].idxmax()
        best_cumulative_index = regime_data["Cumulative Return In Regime"].idxmax()

        summary.loc[best_sharpe_index, "Best By Sharpe"] = True
        summary.loc[best_return_index, "Best By Annualized Return"] = True
        summary.loc[best_cumulative_index, "Best By Cumulative Return"] = True

    return summary


def format_summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    display_df = summary.copy()

    percent_columns = [
        "Mean Monthly Return",
        "Annualized Return",
        "Annualized Volatility",
        "Cumulative Return In Regime",
        "Hit Rate",
        "Best Month",
        "Worst Month",
    ]

    for column in percent_columns:
        display_df[column] = display_df[column].map(lambda x: f"{x:.2%}")

    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].map(lambda x: f"{x:.2f}")

    return display_df


def print_best_strategies(summary: pd.DataFrame) -> None:
    print("\nBest Strategy by Regime")
    print("=" * 100)

    for regime in sorted(summary["Regime"].unique()):
        regime_data = summary[summary["Regime"] == regime]
        description = regime_data["Regime Description"].iloc[0]

        best_sharpe = regime_data.loc[regime_data["Sharpe Ratio"].idxmax()]
        best_return = regime_data.loc[regime_data["Annualized Return"].idxmax()]
        best_cumulative = regime_data.loc[
            regime_data["Cumulative Return In Regime"].idxmax()
        ]

        print(f"\nRegime {regime}: {description}")
        print(f"Best by Sharpe: {best_sharpe['Strategy']}")
        print(f"Best by Annualized Return: {best_return['Strategy']}")
        print(f"Best by Cumulative Return: {best_cumulative['Strategy']}")


def plot_metric_by_regime(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = summary.pivot(
        index="Regime",
        columns="Strategy",
        values=metric,
    )

    ax = pivot.plot(kind="bar", figsize=(12, 6))

    ax.set_title(title)
    ax.set_xlabel("Regime")
    ax.set_ylabel(metric)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")


def main() -> None:
    prices = load_prices()
    returns = calculate_daily_returns(prices)

    transaction_cost_bps = 10

    results = run_all_strategies(
        returns=returns,
        transaction_cost_bps=transaction_cost_bps,
    )

    regime_data, descriptions = build_regime_data(prices)

    monthly_returns = calculate_monthly_strategy_returns(results)

    monthly_returns_with_regimes = align_returns_with_regimes(
        monthly_returns=monthly_returns,
        regime_data=regime_data,
    )

    summary = summarize_strategy_performance_by_regime(
        monthly_returns_with_regimes=monthly_returns_with_regimes,
        descriptions=descriptions,
    )

    summary = add_best_strategy_flags(summary)

    REGIME_STRATEGY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REGIME_STRATEGY_OUTPUT_PATH, index=False)

    print("\nStrategy Performance by Market Regime")
    print("=" * 100)
    print(f"Transaction cost assumption: {transaction_cost_bps} bps per dollar traded")
    print("-" * 100)

    display_summary = format_summary_for_display(summary)

    columns_to_display = [
        "Regime",
        "Regime Description",
        "Strategy",
        "Months",
        "Mean Monthly Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Cumulative Return In Regime",
        "Hit Rate",
        "Best Month",
        "Worst Month",
    ]

    print(display_summary[columns_to_display].to_markdown(index=False))

    print_best_strategies(summary)

    plot_metric_by_regime(
        summary=summary,
        metric="Annualized Return",
        output_path=RETURN_FIGURE_PATH,
        title="Annualized Return by Strategy and Regime",
    )

    plot_metric_by_regime(
        summary=summary,
        metric="Sharpe Ratio",
        output_path=SHARPE_FIGURE_PATH,
        title="Sharpe Ratio by Strategy and Regime",
    )

    plot_metric_by_regime(
        summary=summary,
        metric="Cumulative Return In Regime",
        output_path=CUMULATIVE_FIGURE_PATH,
        title="Cumulative Return by Strategy and Regime",
    )

    print(f"\nSaved regime strategy summary to: {REGIME_STRATEGY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()