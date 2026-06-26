# Final Results Summary

## Project

**Regime-Aware Portfolio Optimization Under Macroeconomic Uncertainty**

This project evaluates whether a regime-aware ETF allocation system can improve risk-adjusted performance compared with static and optimization-based portfolio strategies.

## Final Strategy

The final selected strategy is:

**Regime-Aware 10% Threshold**

Rule:

- Use the previous month's detected market regime.
- Select the corresponding portfolio logic.
- Rebalance monthly.
- Only execute a rebalance if proposed turnover is at least 10%.
- Apply transaction costs of 10 bps per dollar traded.

## Final Performance Table

| Strategy                   | Final Value   | CAGR   | Volatility   |   Sharpe Ratio | Max Drawdown   | Average Rebalance Turnover   | Total Transaction Costs   |
|:---------------------------|:--------------|:-------|:-------------|---------------:|:---------------|:-----------------------------|:--------------------------|
| Equal Weight               | $42,557.43    | 7.85%  | 13.79%       |           0.61 | -40.43%        |                              |                           |
| 60/40                      | $45,890.53    | 8.24%  | 11.05%       |           0.77 | -32.50%        |                              |                           |
| Minimum Volatility         | $29,146.45    | 5.70%  | 6.88%        |           0.84 | -19.80%        |                              |                           |
| Maximum Sharpe             | $50,075.32    | 8.75%  | 10.61%       |           0.84 | -24.18%        |                              |                           |
| Regime-Aware 10% Threshold | $49,075.73    | 8.62%  | 9.22%        |           0.94 | -22.40%        | 86.30%                       | $1,727.23                 |

## Main Findings

- Best final value: **Maximum Sharpe**
- Best Sharpe ratio: **Regime-Aware 10% Threshold**
- Lowest drawdown: **Minimum Volatility**
- Final regime-aware CAGR: **8.62%**
- Final regime-aware Sharpe ratio: **0.94**
- Final regime-aware max drawdown: **-22.40%**

## Latest Final Regime-Aware ETF Weights

|     | 2026-06-26 00:00:00   |
|:----|:----------------------|
| SPY | 59.18%                |
| IEF | 40.82%                |

## Interpretation

The regime-aware strategy improves risk-adjusted performance by adapting portfolio construction to the previous month's market regime. The 10% turnover threshold reduces unnecessary rebalancing while preserving the strategy's ability to respond to meaningful regime changes.

The main tradeoff is that the regime-aware strategy does not always maximize final wealth, but it offers a stronger balance of return, volatility, and drawdown control.

## Generated Figures

- `figures/final_assets/final_portfolio_value_comparison.png`
- `figures/final_assets/final_drawdown_comparison.png`
- `figures/final_assets/final_rolling_sharpe_comparison.png`
- `figures/final_assets/final_regime_aware_weights.png`
