import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_min_vol_weights(
    returns: pd.DataFrame,
    assets: list[str] | None = None,
    max_weight: float = 0.40,
) -> pd.Series:
    """
    Calculate long-only minimum volatility portfolio weights.

    Parameters
    ----------
    returns:
        Historical daily returns used to estimate the covariance matrix.

    assets:
        List of asset tickers.

    max_weight:
        Maximum allocation allowed for any single asset.

    Returns
    -------
    pd.Series
        Optimized portfolio weights.
    """

    if assets is None:
        assets = list(returns.columns)

    returns = returns[assets].dropna()

    num_assets = len(assets)

    if returns.empty or len(returns) < 2:
        return pd.Series(1 / num_assets, index=assets)

    if max_weight * num_assets < 1:
        raise ValueError("max_weight is too restrictive. Weights cannot sum to 1.")

    covariance_matrix = returns.cov().values * 252
    covariance_matrix = np.nan_to_num(covariance_matrix, nan=0.0)

    # Small diagonal adjustment improves numerical stability.
    covariance_matrix += np.eye(num_assets) * 1e-8

    def portfolio_volatility(weights: np.ndarray) -> float:
        variance = weights.T @ covariance_matrix @ weights
        return float(np.sqrt(variance))

    initial_guess = np.repeat(1 / num_assets, num_assets)

    bounds = tuple((0.0, max_weight) for _ in range(num_assets))

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1,
    }

    result = minimize(
        portfolio_volatility,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        return pd.Series(1 / num_assets, index=assets)

    weights = pd.Series(result.x, index=assets)
    weights = weights.clip(lower=0)
    weights = weights / weights.sum()

    return weights