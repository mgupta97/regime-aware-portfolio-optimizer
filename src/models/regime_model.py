import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def fit_kmeans_regimes(
    features: pd.DataFrame,
    n_regimes: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit K-Means clustering to market features and assign regime labels.

    This version is exploratory and uses the full sample.
    Later, we will build a rolling version to avoid look-ahead bias.
    """

    if features.empty:
        raise ValueError("Features data is empty.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=n_regimes,
        random_state=random_state,
        n_init=50,
    )

    regime_labels = model.fit_predict(scaled_features)

    regime_data = features.copy()
    regime_data["Regime"] = regime_labels

    return regime_data


def summarize_regimes(regime_data: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize each regime using average feature values and observation counts.
    """

    if "Regime" not in regime_data.columns:
        raise ValueError("Regime column is missing.")

    feature_columns = [col for col in regime_data.columns if col != "Regime"]

    summary = regime_data.groupby("Regime")[feature_columns].mean()
    summary["Observations"] = regime_data.groupby("Regime").size()

    return summary


def assign_regime_descriptions(summary: pd.DataFrame) -> dict[int, str]:
    """
    Assign simple human-readable descriptions to each regime.

    K-Means labels are arbitrary, so this helps interpret what each cluster means.
    """

    descriptions = {}

    lowest_equity_return_regime = summary["SPY_3M_Return"].idxmin()
    highest_equity_return_regime = summary["SPY_3M_Return"].idxmax()
    highest_vol_regime = summary["SPY_3M_Volatility"].idxmax()
    highest_commodity_regime = summary["DBC_3M_Return"].idxmax()
    highest_bond_regime = summary["TLT_3M_Return"].idxmax()

    for regime in summary.index:
        if regime == highest_vol_regime or regime == lowest_equity_return_regime:
            descriptions[regime] = "Stress / high-volatility equity weakness"
        elif regime == highest_equity_return_regime:
            descriptions[regime] = "Equity growth / risk-on"
        elif regime == highest_commodity_regime:
            descriptions[regime] = "Commodity strength / inflation-like"
        elif regime == highest_bond_regime:
            descriptions[regime] = "Defensive / bond strength"
        else:
            descriptions[regime] = "Mixed / transition regime"

    return descriptions