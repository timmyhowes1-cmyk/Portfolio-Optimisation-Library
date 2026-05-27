import pandas as pd
import numpy as np


def build_params(sector_region: pd.DataFrame, bmk_weights: pd.Series) -> dict:
    sector_tolerance = pd.DataFrame(
        {'lower': -0.05, 'upper': 0.05},
        index=sector_region['sector'].unique(),
    )
    sector_tolerance.loc[['Energy', 'Utilities'], 'upper'] = 0

    return {
        "sector_tolerance":             sector_tolerance,
        "industry_tolerance":           0.025,
        "idio_halflife":               126,
        "factor_halflife":              252,
        "stock_active_weight":          0.02,
        "stock_active_weight_multiple": 20,
        "n_effective_stocks":           [1 / float(np.sum(bmk_weights ** 2)), np.inf],
        "max_te":                       0.035,
        "max_turnover":                 0.3,
    }