from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest.metrics import performance_summary
from src.backtest.run_true_regime_aware_backtest import (
    REGIME_STRATEGY_MAP,
    build_regime_data,
    calculate_daily_returns,
    get_equal_weight_targets,
    get_sixty_forty_targets,
    load_prices,
    maximum_sharpe_weight_function,
    minimum_volatility_weight_function,
)


OUTPUT_PATH = Path("data/processed/turnover_control_experiment.csv")
SIGNALS_OUTPUT_PATH = Path("data/processed/turnover_control_signals.csv")

FINAL_VALUE_FIGURE_PATH = Path("figures/turnover_control_final_value.png")
SHARPE_FIGURE_PATH = Path("figures/turnover_control_sharpe.png")
COST_FIGURE_PATH = Path("figures/turnover_control_transaction_costs.png")
TURNOVER_FIGURE_PATH = Path("figures/turnover_control_average_turnover.png")


EXPERIMENT_CONFIGS = [
    {
        "Experiment": "Monthly Base",
        "Rebalance Frequency": "monthly",
        "Turnover Threshold": 0.00,
        "Persistence Months": 1,
    },
    {
        "Experiment": "Monthly 5% Threshold",
        "Rebalance Frequency": "monthly",
        "Turnover Threshold": 0.05,
        "Persistence Months": 1,
    },
    {
        "Experiment": "Monthly 10% Threshold",
        "Rebalance Frequency": "monthly",
        "Turnover Threshold": 0.10,
        "Persistence Months": 1,
    },
    {
        "Experiment": "Monthly 20% Threshold",
        "Rebalance Frequency": "monthly",
        "Turnover Threshold": 0.20,
        "Persistence Months": 1,
    },
    {
        "Experiment": "Quarterly Base",
        "Rebalance Frequency": "quarterly",
        "Turnover Threshold": 0.00,
        "Persistence Months": 1,
    },
    {
        "Experiment": "Quarterly 5% Threshold",
        "Rebalance Frequency": "quarterly",
        "Turnover Threshold": 0.05,
        "Persistence Months": 1,
    },
    {
        "Experiment": "Monthly 2-Month Regime Confirmation",
        "Rebalance Frequency": "monthly",
        "Turnover Threshold": 0.00,
        "Persistence Months": 2,
    },
    {
        "Experiment": "Monthly 2-Month Confirmation + 5% Threshold",
        "Rebalance Frequency": "monthly",
        "Turnover Threshold": 0.05,
        "Persistence Months": 2,
    },
    {
        "Experiment": "Quarterly 2-Month Confirmation + 5% Threshold",
        "Rebalance Frequency": "quarterly",
        "Turnover Threshold": 0.05,
        "Persistence Months": 2,
    },
]


def get_rebalance_dates(
    returns: pd.DataFrame,
    frequency: str,
) -> set[pd.Timestamp]:
    monthly_first_dates = returns.groupby(returns.index.to_period("M")).head(1).index

    if frequency == "monthly":
        return set(monthly_first_dates)

    if frequency == "quarterly":
        quarterly_first_dates = [
            date for date in monthly_first_dates if date.month in [1, 4, 7, 10]
        ]
        return set(quarterly_first_dates)

    raise ValueError(f"Unknown rebalance frequency: {frequency}")


def find_first_signal_date(
    returns: pd.DataFrame,
    regime_data: pd.DataFrame,
) -> pd.Timestamp:
    regime_months = set(regime_data.index.to_period("M"))

    for date in returns.index:
        previous_month = date.to_period("M") - 1

        if previous_month in regime_months:
            return date

    raise ValueError("No valid signal date found.")


def get_signal_from_regime(
    current_month: pd.Period,
    regime_by_month: pd.Series,
    descriptions: dict[int, str],
    previous_implemented_regime: int | None,
    previous_implemented_strategy: str | None,
    persistence_months: int,
) -> tuple[int | None, str | None, str | None]:
    """
    Select a strategy using prior regime information.

    If persistence_months = 1:
        Use previous month's regime.

    If persistence_months = 2:
        Only accept a regime switch when the previous two months had the same regime.
        Otherwise, keep the previously implemented strategy if one exists.
    """

    previous_month = current_month - 1

    if previous_month not in regime_by_month.index:
        return None, None, None

    previous_regime = int(regime_by_month.loc[previous_month])

    if persistence_months <= 1:
        selected_regime = previous_regime
        selected_strategy = REGIME_STRATEGY_MAP[selected_regime]
        description = descriptions.get(selected_regime, "Unknown")
        return selected_regime, selected_strategy, description

    two_months_ago = current_month - 2

    if two_months_ago in regime_by_month.index:
        two_months_ago_regime = int(regime_by_month.loc[two_months_ago])

        if previous_regime == two_months_ago_regime:
            selected_regime = previous_regime
            selected_strategy = REGIME_STRATEGY_MAP[selected_regime]
            description = descriptions.get(selected_regime, "Unknown")
            return selected_regime, selected_strategy, description

    if previous_implemented_strategy is not None:
        description = descriptions.get(previous_implemented_regime, "Unknown")
        return (
            previous_implemented_regime,
            previous_implemented_strategy,
            description,
        )

    selected_regime = previous_regime
    selected_strategy = REGIME_STRATEGY_MAP[selected_regime]
    description = descriptions.get(selected_regime, "Unknown")

    return selected_regime, selected_strategy, description


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


def validate_weights(weights: pd.Series, assets: list[str]) -> pd.Series:
    weights = weights.reindex(assets).fillna(0.0)

    if (weights < 0).any():
        raise ValueError("Negative weights are not allowed.")

    weight_sum = weights.sum()

    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"Weights must sum to 1. Current sum: {weight_sum:.4f}")

    return weights


def run_turnover_controlled_backtest(
    returns: pd.DataFrame,
    regime_data: pd.DataFrame,
    descriptions: dict[int, str],
    experiment_name: str,
    rebalance_frequency: str,
    turnover_threshold: float,
    persistence_months: int,
    starting_value: float = 10_000,
    transaction_cost_bps: float = 10,
    lookback_days: int = 252,
    min_observations: int = 60,
) -> dict:
    if returns.empty:
        raise ValueError("Returns data is empty.")

    assets = list(returns.columns)
    transaction_cost_rate = transaction_cost_bps / 10_000

    regime_by_month = regime_data["Regime"].copy()
    regime_by_month.index = regime_by_month.index.to_period("M")

    rebalance_dates = get_rebalance_dates(
        returns=returns,
        frequency=rebalance_frequency,
    )

    start_date = find_first_signal_date(returns, regime_data)

    portfolio_value = starting_value
    previous_portfolio_value = np.nan

    current_weights = pd.Series(0.0, index=assets)
    implemented_strategy = None
    implemented_regime = None

    dates = []
    portfolio_values = []
    portfolio_returns = []
    weight_history = []
    turnover_history = []
    transaction_cost_history = []
    signal_history = []

    for row_number, (date, daily_returns) in enumerate(returns.iterrows()):
        if date < start_date:
            continue

        proposed_strategy = None
        proposed_regime = None
        regime_description = None
        executed_rebalance = False
        skipped_due_to_threshold = False
        proposed_turnover = 0.0
        actual_turnover = 0.0
        transaction_cost = 0.0

        if date in rebalance_dates or current_weights.sum() == 0:
            current_month = date.to_period("M")

            proposed_regime, proposed_strategy, regime_description = get_signal_from_regime(
                current_month=current_month,
                regime_by_month=regime_by_month,
                descriptions=descriptions,
                previous_implemented_regime=implemented_regime,
                previous_implemented_strategy=implemented_strategy,
                persistence_months=persistence_months,
            )

            if proposed_strategy is None:
                continue

            lookback_start = max(0, row_number - lookback_days)
            historical_returns = returns.iloc[lookback_start:row_number]

            target_weights = get_target_weights_for_strategy(
                selected_strategy=proposed_strategy,
                historical_returns=historical_returns,
                assets=assets,
                min_observations=min_observations,
            )

            target_weights = validate_weights(target_weights, assets)

            proposed_turnover = float((target_weights - current_weights).abs().sum())

            should_rebalance = (
                current_weights.sum() == 0
                or proposed_turnover >= turnover_threshold
            )

            if should_rebalance:
                actual_turnover = proposed_turnover
                transaction_cost = portfolio_value * actual_turnover * transaction_cost_rate

                portfolio_value -= transaction_cost
                current_weights = target_weights.copy()

                implemented_strategy = proposed_strategy
                implemented_regime = proposed_regime
                executed_rebalance = True
            else:
                skipped_due_to_threshold = True

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

        dates.append(date)
        portfolio_values.append(portfolio_value)
        portfolio_returns.append(realized_return)
        weight_history.append(current_weights.copy())
        turnover_history.append(actual_turnover)
        transaction_cost_history.append(transaction_cost)

        signal_history.append(
            {
                "Date": date,
                "Experiment": experiment_name,
                "Proposed Regime": proposed_regime,
                "Regime Description": regime_description,
                "Proposed Strategy": proposed_strategy,
                "Implemented Regime": implemented_regime,
                "Implemented Strategy": implemented_strategy,
                "Proposed Turnover": proposed_turnover,
                "Actual Turnover": actual_turnover,
                "Transaction Cost": transaction_cost,
                "Executed Rebalance": executed_rebalance,
                "Skipped Due To Threshold": skipped_due_to_threshold,
            }
        )

    if not portfolio_values:
        raise ValueError(f"No portfolio values generated for {experiment_name}.")

    portfolio_values = pd.Series(
        portfolio_values,
        index=pd.DatetimeIndex(dates),
        name=experiment_name,
    )

    portfolio_returns = pd.Series(
        portfolio_returns,
        index=pd.DatetimeIndex(dates),
        name=experiment_name,
    ).dropna()

    weights = pd.DataFrame(
        weight_history,
        index=pd.DatetimeIndex(dates),
        columns=assets,
    )

    turnover = pd.Series(
        turnover_history,
        index=pd.DatetimeIndex(dates),
        name="Turnover",
    )

    transaction_costs = pd.Series(
        transaction_cost_history,
        index=pd.DatetimeIndex(dates),
        name="Transaction Costs",
    )

    signals = pd.DataFrame(signal_history).set_index("Date")

    return {
        "portfolio_values": portfolio_values,
        "portfolio_returns": portfolio_returns,
        "weights": weights,
        "turnover": turnover,
        "transaction_costs": transaction_costs,
        "signals": signals,
    }


def summarize_experiment(
    experiment_name: str,
    result: dict,
    rebalance_frequency: str,
    turnover_threshold: float,
    persistence_months: int,
) -> dict:
    summary = performance_summary(
        portfolio_returns=result["portfolio_returns"],
        portfolio_name=experiment_name,
        portfolio_values=result["portfolio_values"],
        turnover=result["turnover"],
        transaction_costs=result["transaction_costs"],
    )

    signals = result["signals"]
    rebalance_signals = signals[
        (signals["Executed Rebalance"]) | (signals["Skipped Due To Threshold"])
    ]

    summary["Rebalance Frequency"] = rebalance_frequency
    summary["Turnover Threshold"] = turnover_threshold
    summary["Persistence Months"] = persistence_months
    summary["Executed Rebalances"] = int(signals["Executed Rebalance"].sum())
    summary["Skipped Rebalances"] = int(signals["Skipped Due To Threshold"].sum())
    summary["Rebalance Opportunities"] = len(rebalance_signals)

    return summary


def run_experiments(transaction_cost_bps: float = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_prices()
    returns = calculate_daily_returns(prices)

    regime_data, descriptions = build_regime_data(prices)

    summaries = []
    all_signals = []

    for config in EXPERIMENT_CONFIGS:
        experiment_name = config["Experiment"]

        print(f"\nRunning experiment: {experiment_name}")

        result = run_turnover_controlled_backtest(
            returns=returns,
            regime_data=regime_data,
            descriptions=descriptions,
            experiment_name=experiment_name,
            rebalance_frequency=config["Rebalance Frequency"],
            turnover_threshold=config["Turnover Threshold"],
            persistence_months=config["Persistence Months"],
            starting_value=10_000,
            transaction_cost_bps=transaction_cost_bps,
            lookback_days=252,
            min_observations=60,
        )

        summary = summarize_experiment(
            experiment_name=experiment_name,
            result=result,
            rebalance_frequency=config["Rebalance Frequency"],
            turnover_threshold=config["Turnover Threshold"],
            persistence_months=config["Persistence Months"],
        )

        summaries.append(summary)

        signals = result["signals"].copy()
        all_signals.append(signals)

    summary_df = pd.DataFrame(summaries)
    signals_df = pd.concat(all_signals)

    return summary_df, signals_df


def format_summary_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    display_df = summary_df.copy()

    display_df["Final Value"] = display_df["Final Value"].map(lambda x: f"${x:,.2f}")
    display_df["CAGR"] = display_df["CAGR"].map(lambda x: f"{x:.2%}")
    display_df["Volatility"] = display_df["Volatility"].map(lambda x: f"{x:.2%}")
    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].map(lambda x: f"{x:.2%}")
    display_df["Turnover Threshold"] = display_df["Turnover Threshold"].map(
        lambda x: f"{x:.2%}"
    )

    if "Average Rebalance Turnover" in display_df.columns:
        display_df["Average Rebalance Turnover"] = display_df[
            "Average Rebalance Turnover"
        ].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")

    if "Total Transaction Costs" in display_df.columns:
        display_df["Total Transaction Costs"] = display_df[
            "Total Transaction Costs"
        ].map(lambda x: "" if pd.isna(x) else f"${x:,.2f}")

    return display_df


def plot_bar_metric(
    summary_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = summary_df.sort_values(metric, ascending=False)

    ax = plot_df.plot(
        x="Strategy",
        y=metric,
        kind="bar",
        legend=False,
        figsize=(12, 6),
    )

    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {output_path}")


def main() -> None:
    transaction_cost_bps = 10

    summary_df, signals_df = run_experiments(
        transaction_cost_bps=transaction_cost_bps,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)
    signals_df.to_csv(SIGNALS_OUTPUT_PATH)

    print("\nTurnover Control Experiment Results")
    print("=" * 100)
    print(f"Transaction cost assumption: {transaction_cost_bps} bps per dollar traded")
    print("-" * 100)

    columns_to_display = [
        "Strategy",
        "Final Value",
        "CAGR",
        "Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Average Rebalance Turnover",
        "Total Transaction Costs",
        "Rebalance Frequency",
        "Turnover Threshold",
        "Persistence Months",
        "Executed Rebalances",
        "Skipped Rebalances",
    ]

    display_df = format_summary_for_display(summary_df)
    print(display_df[columns_to_display].to_markdown(index=False))

    best_by_sharpe = summary_df.loc[summary_df["Sharpe Ratio"].idxmax()]
    best_by_final_value = summary_df.loc[summary_df["Final Value"].idxmax()]
    lowest_cost = summary_df.loc[summary_df["Total Transaction Costs"].idxmin()]

    print("\nBest Turnover-Control Results")
    print("=" * 100)
    print(f"Best Sharpe: {best_by_sharpe['Strategy']} ({best_by_sharpe['Sharpe Ratio']:.2f})")
    print(
        f"Best Final Value: {best_by_final_value['Strategy']} "
        f"(${best_by_final_value['Final Value']:,.2f})"
    )
    print(
        f"Lowest Transaction Costs: {lowest_cost['Strategy']} "
        f"(${lowest_cost['Total Transaction Costs']:,.2f})"
    )

    plot_bar_metric(
        summary_df=summary_df,
        metric="Final Value",
        output_path=FINAL_VALUE_FIGURE_PATH,
        title="Final Value by Turnover-Control Experiment",
        ylabel="Final Value ($)",
    )

    plot_bar_metric(
        summary_df=summary_df,
        metric="Sharpe Ratio",
        output_path=SHARPE_FIGURE_PATH,
        title="Sharpe Ratio by Turnover-Control Experiment",
        ylabel="Sharpe Ratio",
    )

    plot_bar_metric(
        summary_df=summary_df,
        metric="Total Transaction Costs",
        output_path=COST_FIGURE_PATH,
        title="Transaction Costs by Turnover-Control Experiment",
        ylabel="Total Transaction Costs ($)",
    )

    plot_bar_metric(
        summary_df=summary_df,
        metric="Average Rebalance Turnover",
        output_path=TURNOVER_FIGURE_PATH,
        title="Average Rebalance Turnover by Experiment",
        ylabel="Average Rebalance Turnover",
    )

    print(f"\nSaved turnover-control results to: {OUTPUT_PATH}")
    print(f"Saved turnover-control signals to: {SIGNALS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()