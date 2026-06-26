from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest.metrics import performance_summary
from src.backtest.run_true_regime_aware_backtest import (
    build_regime_data,
    calculate_daily_returns,
    calculate_strategy_returns_from_results,
    load_prices,
    run_baseline_strategies,
)
from src.analysis.turnover_control_experiment import run_turnover_controlled_backtest


OUTPUT_DIR = Path("reports/final_assets")
FIGURE_DIR = Path("figures/final_assets")

FINAL_TABLE_PATH = OUTPUT_DIR / "final_performance_table.csv"
FINAL_MARKDOWN_TABLE_PATH = OUTPUT_DIR / "final_performance_table.md"
FINAL_SUMMARY_PATH = OUTPUT_DIR / "final_results_summary.md"
FINAL_WEIGHTS_PATH = OUTPUT_DIR / "final_regime_aware_weights.csv"
FINAL_RETURNS_PATH = OUTPUT_DIR / "final_strategy_returns.csv"

PORTFOLIO_VALUE_FIGURE_PATH = FIGURE_DIR / "final_portfolio_value_comparison.png"
DRAWDOWN_FIGURE_PATH = FIGURE_DIR / "final_drawdown_comparison.png"
ROLLING_SHARPE_FIGURE_PATH = FIGURE_DIR / "final_rolling_sharpe_comparison.png"
WEIGHTS_FIGURE_PATH = FIGURE_DIR / "final_regime_aware_weights.png"

TRADING_DAYS = 252
STARTING_VALUE = 10_000

FINAL_STRATEGY_NAME = "Regime-Aware 10% Threshold"


def calculate_portfolio_values(
    returns: pd.Series,
    starting_value: float = STARTING_VALUE,
) -> pd.Series:
    return starting_value * (1 + returns).cumprod()


def calculate_drawdown(portfolio_values: pd.Series) -> pd.Series:
    running_max = portfolio_values.cummax()
    return portfolio_values / running_max - 1


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    rolling_mean = returns.rolling(window).mean()
    rolling_volatility = returns.rolling(window).std()

    rolling_sharpe = rolling_mean / rolling_volatility * np.sqrt(TRADING_DAYS)

    return rolling_sharpe


def build_final_strategy(
    transaction_cost_bps: float = 10,
) -> tuple[pd.DataFrame, dict]:
    prices = load_prices()
    returns = calculate_daily_returns(prices)

    regime_data, descriptions = build_regime_data(prices)

    baseline_results = run_baseline_strategies(
        returns=returns,
        transaction_cost_bps=transaction_cost_bps,
    )

    baseline_returns = calculate_strategy_returns_from_results(baseline_results)

    final_result = run_turnover_controlled_backtest(
        returns=returns,
        regime_data=regime_data,
        descriptions=descriptions,
        experiment_name=FINAL_STRATEGY_NAME,
        rebalance_frequency="monthly",
        turnover_threshold=0.10,
        persistence_months=1,
        starting_value=STARTING_VALUE,
        transaction_cost_bps=transaction_cost_bps,
        lookback_days=252,
        min_observations=60,
    )

    final_returns = final_result["portfolio_returns"].rename(FINAL_STRATEGY_NAME)

    comparison_returns = baseline_returns.copy()
    comparison_returns[FINAL_STRATEGY_NAME] = final_returns

    comparison_returns = comparison_returns.loc[final_returns.index]
    comparison_returns = comparison_returns.dropna(how="any")

    return comparison_returns, final_result


def summarize_final_results(
    comparison_returns: pd.DataFrame,
    final_result: dict,
) -> pd.DataFrame:
    summaries = []

    for strategy in comparison_returns.columns:
        strategy_returns = comparison_returns[strategy].dropna()
        strategy_values = calculate_portfolio_values(strategy_returns)

        if strategy == FINAL_STRATEGY_NAME:
            summary = performance_summary(
                portfolio_returns=strategy_returns,
                portfolio_name=strategy,
                portfolio_values=strategy_values,
                turnover=final_result["turnover"].loc[strategy_returns.index],
                transaction_costs=final_result["transaction_costs"].loc[
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

    summary_df = pd.DataFrame(summaries)

    return summary_df


def format_summary_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
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


def plot_portfolio_value_comparison(comparison_returns: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for strategy in comparison_returns.columns:
        portfolio_values = calculate_portfolio_values(comparison_returns[strategy])
        plt.plot(portfolio_values.index, portfolio_values, label=strategy)

    plt.title("Final Strategy Comparison: Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PORTFOLIO_VALUE_FIGURE_PATH, dpi=300)
    plt.close()

    print(f"Saved portfolio value chart to: {PORTFOLIO_VALUE_FIGURE_PATH}")


def plot_drawdown_comparison(comparison_returns: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for strategy in comparison_returns.columns:
        portfolio_values = calculate_portfolio_values(comparison_returns[strategy])
        drawdown = calculate_drawdown(portfolio_values)
        plt.plot(drawdown.index, drawdown, label=strategy)

    plt.title("Final Strategy Comparison: Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DRAWDOWN_FIGURE_PATH, dpi=300)
    plt.close()

    print(f"Saved drawdown chart to: {DRAWDOWN_FIGURE_PATH}")


def plot_rolling_sharpe_comparison(comparison_returns: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for strategy in comparison_returns.columns:
        rolling_sharpe = calculate_rolling_sharpe(comparison_returns[strategy])
        plt.plot(rolling_sharpe.index, rolling_sharpe, label=strategy)

    plt.title("Final Strategy Comparison: Rolling 252-Day Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROLLING_SHARPE_FIGURE_PATH, dpi=300)
    plt.close()

    print(f"Saved rolling Sharpe chart to: {ROLLING_SHARPE_FIGURE_PATH}")


def plot_final_weights(final_result: dict) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    weights = final_result["weights"].copy()
    monthly_weights = weights.resample("ME").last()

    ax = monthly_weights.plot.area(figsize=(12, 6))

    ax.set_title("Final Regime-Aware Strategy ETF Weights")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Weight")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(WEIGHTS_FIGURE_PATH, dpi=300)
    plt.close()

    print(f"Saved final weights chart to: {WEIGHTS_FIGURE_PATH}")


def save_final_markdown_summary(
    summary_df: pd.DataFrame,
    final_result: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    display_df = format_summary_for_display(summary_df)

    final_strategy_row = summary_df[
        summary_df["Strategy"] == FINAL_STRATEGY_NAME
    ].iloc[0]

    best_final_value_row = summary_df.loc[summary_df["Final Value"].idxmax()]
    best_sharpe_row = summary_df.loc[summary_df["Sharpe Ratio"].idxmax()]
    lowest_drawdown_row = summary_df.loc[summary_df["Max Drawdown"].idxmax()]

    latest_weights = final_result["weights"].iloc[-1]
    latest_weights = latest_weights[latest_weights > 0.01].sort_values(ascending=False)
    latest_weights_display = latest_weights.map(lambda x: f"{x:.2%}")

    markdown = f"""# Final Results Summary

## Project

**Regime-Aware Portfolio Optimization Under Macroeconomic Uncertainty**

This project evaluates whether a regime-aware ETF allocation system can improve risk-adjusted performance compared with static and optimization-based portfolio strategies.

## Final Strategy

The final selected strategy is:

**{FINAL_STRATEGY_NAME}**

Rule:

- Use the previous month's detected market regime.
- Select the corresponding portfolio logic.
- Rebalance monthly.
- Only execute a rebalance if proposed turnover is at least 10%.
- Apply transaction costs of 10 bps per dollar traded.

## Final Performance Table

{display_df.to_markdown(index=False)}

## Main Findings

- Best final value: **{best_final_value_row["Strategy"]}**
- Best Sharpe ratio: **{best_sharpe_row["Strategy"]}**
- Lowest drawdown: **{lowest_drawdown_row["Strategy"]}**
- Final regime-aware CAGR: **{final_strategy_row["CAGR"]:.2%}**
- Final regime-aware Sharpe ratio: **{final_strategy_row["Sharpe Ratio"]:.2f}**
- Final regime-aware max drawdown: **{final_strategy_row["Max Drawdown"]:.2%}**

## Latest Final Regime-Aware ETF Weights

{latest_weights_display.to_markdown()}

## Interpretation

The regime-aware strategy improves risk-adjusted performance by adapting portfolio construction to the previous month's market regime. The 10% turnover threshold reduces unnecessary rebalancing while preserving the strategy's ability to respond to meaningful regime changes.

The main tradeoff is that the regime-aware strategy does not always maximize final wealth, but it offers a stronger balance of return, volatility, and drawdown control.

## Generated Figures

- `figures/final_assets/final_portfolio_value_comparison.png`
- `figures/final_assets/final_drawdown_comparison.png`
- `figures/final_assets/final_rolling_sharpe_comparison.png`
- `figures/final_assets/final_regime_aware_weights.png`
"""

    FINAL_SUMMARY_PATH.write_text(markdown)

    print(f"Saved final markdown summary to: {FINAL_SUMMARY_PATH}")


def save_outputs(
    summary_df: pd.DataFrame,
    comparison_returns: pd.DataFrame,
    final_result: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(FINAL_TABLE_PATH, index=False)

    display_df = format_summary_for_display(summary_df)
    FINAL_MARKDOWN_TABLE_PATH.write_text(display_df.to_markdown(index=False))

    comparison_returns.to_csv(FINAL_RETURNS_PATH)
    final_result["weights"].to_csv(FINAL_WEIGHTS_PATH)

    print(f"Saved final performance table to: {FINAL_TABLE_PATH}")
    print(f"Saved final markdown table to: {FINAL_MARKDOWN_TABLE_PATH}")
    print(f"Saved final strategy returns to: {FINAL_RETURNS_PATH}")
    print(f"Saved final weights to: {FINAL_WEIGHTS_PATH}")


def print_latest_final_weights(final_result: dict) -> None:
    latest_weights = final_result["weights"].iloc[-1]
    latest_weights = latest_weights[latest_weights > 0.01].sort_values(ascending=False)

    print("\nLatest Final Regime-Aware ETF Weights")
    print("=" * 100)

    display_weights = latest_weights.map(lambda x: f"{x:.2%}")
    print(display_weights.to_markdown())


def main() -> None:
    transaction_cost_bps = 10

    comparison_returns, final_result = build_final_strategy(
        transaction_cost_bps=transaction_cost_bps,
    )

    summary_df = summarize_final_results(
        comparison_returns=comparison_returns,
        final_result=final_result,
    )

    print("\nFinal Report Asset Generation")
    print("=" * 100)
    print(f"Final selected strategy: {FINAL_STRATEGY_NAME}")
    print(f"Transaction cost assumption: {transaction_cost_bps} bps per dollar traded")
    print("-" * 100)

    display_df = format_summary_for_display(summary_df)
    print(display_df.to_markdown(index=False))

    print_latest_final_weights(final_result)

    plot_portfolio_value_comparison(comparison_returns)
    plot_drawdown_comparison(comparison_returns)
    plot_rolling_sharpe_comparison(comparison_returns)
    plot_final_weights(final_result)

    save_outputs(
        summary_df=summary_df,
        comparison_returns=comparison_returns,
        final_result=final_result,
    )

    save_final_markdown_summary(
        summary_df=summary_df,
        final_result=final_result,
    )


if __name__ == "__main__":
    main()