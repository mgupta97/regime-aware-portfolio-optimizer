from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest.run_true_regime_aware_backtest import (
    build_comparison_returns,
    build_regime_data,
    calculate_daily_returns,
    calculate_strategy_returns_from_results,
    load_prices,
    run_baseline_strategies,
    run_true_regime_aware_backtest,
    summarize_comparison,
)


OUTPUT_PATH = Path("data/processed/transaction_cost_sensitivity.csv")
TRUE_STRATEGY_OUTPUT_PATH = Path(
    "data/processed/true_regime_aware_transaction_cost_sensitivity.csv"
)

FINAL_VALUE_FIGURE_PATH = Path("figures/transaction_cost_sensitivity_final_value.png")
CAGR_FIGURE_PATH = Path("figures/transaction_cost_sensitivity_cagr.png")
SHARPE_FIGURE_PATH = Path("figures/transaction_cost_sensitivity_sharpe.png")
DRAWDOWN_FIGURE_PATH = Path("figures/transaction_cost_sensitivity_drawdown.png")


TRANSACTION_COST_SCENARIOS = [0, 5, 10, 25, 50]


def run_sensitivity_analysis(
    transaction_cost_scenarios: list[int],
) -> pd.DataFrame:
    prices = load_prices()
    returns = calculate_daily_returns(prices)

    regime_data, descriptions = build_regime_data(prices)

    all_summaries = []

    for transaction_cost_bps in transaction_cost_scenarios:
        print(f"\nRunning transaction cost scenario: {transaction_cost_bps} bps")

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

        summary_df["Transaction Cost Bps"] = transaction_cost_bps

        all_summaries.append(summary_df)

    sensitivity_df = pd.concat(all_summaries, ignore_index=True)

    return sensitivity_df


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


def plot_metric_by_transaction_cost(
    summary_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = summary_df.pivot(
        index="Transaction Cost Bps",
        columns="Strategy",
        values=metric,
    )

    ax = pivot.plot(marker="o", figsize=(12, 6))

    ax.set_title(title)
    ax.set_xlabel("Transaction Cost Assumption (bps)")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")


def print_true_strategy_sensitivity(true_strategy_df: pd.DataFrame) -> None:
    print("\nTrue Regime-Aware Strategy: Transaction Cost Sensitivity")
    print("=" * 100)

    columns_to_display = [
        "Transaction Cost Bps",
        "Final Value",
        "CAGR",
        "Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Average Rebalance Turnover",
        "Total Transaction Costs",
    ]

    display_df = format_summary_for_display(true_strategy_df)
    print(display_df[columns_to_display].to_markdown(index=False))


def print_all_strategy_sensitivity(summary_df: pd.DataFrame) -> None:
    print("\nAll Strategies: Transaction Cost Sensitivity")
    print("=" * 100)

    columns_to_display = [
        "Transaction Cost Bps",
        "Strategy",
        "Final Value",
        "CAGR",
        "Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
    ]

    display_df = format_summary_for_display(summary_df)
    print(display_df[columns_to_display].to_markdown(index=False))


def main() -> None:
    sensitivity_df = run_sensitivity_analysis(
        transaction_cost_scenarios=TRANSACTION_COST_SCENARIOS,
    )

    true_strategy_df = sensitivity_df[
        sensitivity_df["Strategy"] == "True Regime-Aware Strategy"
    ].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sensitivity_df.to_csv(OUTPUT_PATH, index=False)
    true_strategy_df.to_csv(TRUE_STRATEGY_OUTPUT_PATH, index=False)

    print_true_strategy_sensitivity(true_strategy_df)
    print_all_strategy_sensitivity(sensitivity_df)

    plot_metric_by_transaction_cost(
        summary_df=sensitivity_df,
        metric="Final Value",
        output_path=FINAL_VALUE_FIGURE_PATH,
        title="Final Portfolio Value vs Transaction Cost Assumption",
        ylabel="Final Portfolio Value ($)",
    )

    plot_metric_by_transaction_cost(
        summary_df=sensitivity_df,
        metric="CAGR",
        output_path=CAGR_FIGURE_PATH,
        title="CAGR vs Transaction Cost Assumption",
        ylabel="CAGR",
    )

    plot_metric_by_transaction_cost(
        summary_df=sensitivity_df,
        metric="Sharpe Ratio",
        output_path=SHARPE_FIGURE_PATH,
        title="Sharpe Ratio vs Transaction Cost Assumption",
        ylabel="Sharpe Ratio",
    )

    plot_metric_by_transaction_cost(
        summary_df=sensitivity_df,
        metric="Max Drawdown",
        output_path=DRAWDOWN_FIGURE_PATH,
        title="Max Drawdown vs Transaction Cost Assumption",
        ylabel="Max Drawdown",
    )

    print(f"\nSaved full sensitivity results to: {OUTPUT_PATH}")
    print(f"Saved true strategy sensitivity results to: {TRUE_STRATEGY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()