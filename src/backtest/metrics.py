import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_cagr(portfolio_values: pd.Series) -> float:
    """
    Compound Annual Growth Rate.
    """
    start_value = portfolio_values.iloc[0]
    end_value = portfolio_values.iloc[-1]

    num_days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
    num_years = num_days / 365.25

    if num_years <= 0:
        return np.nan

    return (end_value / start_value) ** (1 / num_years) - 1


def calculate_annualized_volatility(returns: pd.Series) -> float:
    """
    Annualized volatility from daily returns.
    """
    return returns.std() * np.sqrt(TRADING_DAYS)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio using daily returns.
    Assumes risk_free_rate is annualized.
    """
    daily_rf = risk_free_rate / TRADING_DAYS
    excess_returns = returns - daily_rf

    volatility = excess_returns.std()

    if volatility == 0:
        return np.nan

    return excess_returns.mean() / volatility * np.sqrt(TRADING_DAYS)


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Maximum drawdown from portfolio value series.
    """
    running_max = portfolio_values.cummax()
    drawdown = portfolio_values / running_max - 1
    return drawdown.min()


def performance_summary(
    portfolio_returns: pd.Series,
    portfolio_name: str,
    starting_value: float = 10_000,
) -> dict:
    """
    Create performance summary for one portfolio strategy.
    """

    portfolio_values = starting_value * (1 + portfolio_returns).cumprod()

    return {
        "Strategy": portfolio_name,
        "Final Value": portfolio_values.iloc[-1],
        "CAGR": calculate_cagr(portfolio_values),
        "Volatility": calculate_annualized_volatility(portfolio_returns),
        "Sharpe Ratio": calculate_sharpe_ratio(portfolio_returns),
        "Max Drawdown": calculate_max_drawdown(portfolio_values),
    }