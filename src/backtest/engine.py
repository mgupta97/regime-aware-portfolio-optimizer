from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    strategy_name: str
    portfolio_values: pd.Series
    portfolio_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series


def get_monthly_rebalance_dates(returns: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Returns the first trading day and then the last available trading day of each month.
    """

    month_end_dates = returns.groupby(returns.index.to_period("M")).tail(1).index
    first_date = returns.index[0]

    rebalance_dates = pd.DatetimeIndex([first_date]).union(pd.DatetimeIndex(month_end_dates))

    return rebalance_dates


def validate_weights(weights: pd.Series, assets: list[str]) -> pd.Series:
    """
    Ensures weights are aligned to asset columns, have no missing values,
    and sum to 1.
    """

    weights = weights.reindex(assets).fillna(0.0)

    weight_sum = weights.sum()

    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"Weights must sum to 1. Current sum: {weight_sum:.4f}")

    if (weights < 0).any():
        raise ValueError("This engine currently supports long-only portfolios only.")

    return weights


def run_rebalanced_backtest(
    returns: pd.DataFrame,
    target_weights: pd.Series,
    strategy_name: str,
    starting_value: float = 10_000,
    transaction_cost_bps: float = 10,
) -> BacktestResult:
    """
    Runs a monthly rebalanced portfolio backtest.

    Parameters
    ----------
    returns:
        Daily asset returns.

    target_weights:
        Portfolio target weights.

    strategy_name:
        Name of the strategy.

    starting_value:
        Initial portfolio value.

    transaction_cost_bps:
        Transaction cost in basis points.
        Example: 10 bps = 0.10% per dollar traded.

    Returns
    -------
    BacktestResult
    """

    if returns.empty:
        raise ValueError("Returns data is empty.")

    assets = list(returns.columns)
    target_weights = validate_weights(target_weights, assets)

    transaction_cost_rate = transaction_cost_bps / 10_000

    rebalance_dates = get_monthly_rebalance_dates(returns)

    portfolio_value = starting_value

    current_weights = pd.Series(0.0, index=assets)

    portfolio_values = []
    weight_history = []
    turnover_history = []
    transaction_cost_history = []

    for date, daily_returns in returns.iterrows():
        turnover = 0.0
        transaction_cost = 0.0

        if date in rebalance_dates:
            turnover = float((target_weights - current_weights).abs().sum())
            transaction_cost = portfolio_value * turnover * transaction_cost_rate

            portfolio_value -= transaction_cost
            current_weights = target_weights.copy()

        daily_portfolio_return = float(current_weights @ daily_returns)

        portfolio_value *= 1 + daily_portfolio_return

        # After returns occur, weights drift naturally.
        updated_asset_values = current_weights * (1 + daily_returns)

        if updated_asset_values.sum() != 0:
            current_weights = updated_asset_values / updated_asset_values.sum()

        portfolio_values.append(portfolio_value)
        weight_history.append(current_weights.copy())
        turnover_history.append(turnover)
        transaction_cost_history.append(transaction_cost)

    portfolio_values = pd.Series(
        portfolio_values,
        index=returns.index,
        name=strategy_name,
    )

    portfolio_returns = portfolio_values.pct_change().dropna()

    weights = pd.DataFrame(
        weight_history,
        index=returns.index,
        columns=assets,
    )

    turnover = pd.Series(
        turnover_history,
        index=returns.index,
        name="Turnover",
    )

    transaction_costs = pd.Series(
        transaction_cost_history,
        index=returns.index,
        name="Transaction Costs",
    )

    return BacktestResult(
        strategy_name=strategy_name,
        portfolio_values=portfolio_values,
        portfolio_returns=portfolio_returns,
        weights=weights,
        turnover=turnover,
        transaction_costs=transaction_costs,
    )

def run_dynamic_rebalanced_backtest(
    returns: pd.DataFrame,
    weight_function: Callable[[pd.DataFrame, list[str]], pd.Series],
    strategy_name: str,
    lookback_days: int = 252,
    starting_value: float = 10_000,
    transaction_cost_bps: float = 10,
    min_observations: int = 60,
) -> BacktestResult:
    """
    Runs a monthly rebalanced portfolio backtest where weights are recalculated
    at every rebalance date using only historical data.

    This avoids look-ahead bias because the optimizer does not see future returns.
    """

    if returns.empty:
        raise ValueError("Returns data is empty.")

    assets = list(returns.columns)
    num_assets = len(assets)

    transaction_cost_rate = transaction_cost_bps / 10_000

    rebalance_dates = set(get_monthly_rebalance_dates(returns))

    portfolio_value = starting_value
    current_weights = pd.Series(0.0, index=assets)

    portfolio_values = []
    weight_history = []
    turnover_history = []
    transaction_cost_history = []

    for row_number, (date, daily_returns) in enumerate(returns.iterrows()):
        turnover = 0.0
        transaction_cost = 0.0

        if date in rebalance_dates:
            lookback_start = max(0, row_number - lookback_days)
            historical_returns = returns.iloc[lookback_start:row_number]

            if len(historical_returns) < min_observations:
                target_weights = pd.Series(1 / num_assets, index=assets)
            else:
                target_weights = weight_function(historical_returns, assets)

            target_weights = validate_weights(target_weights, assets)

            turnover = float((target_weights - current_weights).abs().sum())
            transaction_cost = portfolio_value * turnover * transaction_cost_rate

            portfolio_value -= transaction_cost
            current_weights = target_weights.copy()

        daily_portfolio_return = float(current_weights @ daily_returns)
        portfolio_value *= 1 + daily_portfolio_return

        updated_asset_values = current_weights * (1 + daily_returns)

        if updated_asset_values.sum() != 0:
            current_weights = updated_asset_values / updated_asset_values.sum()

        portfolio_values.append(portfolio_value)
        weight_history.append(current_weights.copy())
        turnover_history.append(turnover)
        transaction_cost_history.append(transaction_cost)

    portfolio_values = pd.Series(
        portfolio_values,
        index=returns.index,
        name=strategy_name,
    )

    portfolio_returns = portfolio_values.pct_change().dropna()

    weights = pd.DataFrame(
        weight_history,
        index=returns.index,
        columns=assets,
    )

    turnover = pd.Series(
        turnover_history,
        index=returns.index,
        name="Turnover",
    )

    transaction_costs = pd.Series(
        transaction_cost_history,
        index=returns.index,
        name="Transaction Costs",
    )

    return BacktestResult(
        strategy_name=strategy_name,
        portfolio_values=portfolio_values,
        portfolio_returns=portfolio_returns,
        weights=weights,
        turnover=turnover,
        transaction_costs=transaction_costs,
    )