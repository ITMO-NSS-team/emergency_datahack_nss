import numpy as np
import pandas as pd

STATION_COEFFS = {
    3019: 185.35707752426708,
    3027: 1223.8071616577856,
    3028: 1357.4062812989373,
    3029: 1520.7730161870682,
    3030: 1765.9217904996142,
    3035: 765.3703832632036,
    3041: 443.5766934006718,
    3045: 579.1353554017562,
    3050: 612.0471238561079,
    3230: 516.6669876251401
}


def rowwise_nse(row):
    station_id = row.station_id
    station_coeff = STATION_COEFFS[station_id]

    actual = row['delta_stage_max_actual']
    predicted = row['delta_stage_max_predicted']
    return np.divide(np.square(np.subtract(predicted, actual)),
                     station_coeff)


def score(actual_df, predicted_df):
    merged = pd.merge(
        left=predicted_df,
        right=actual_df,
        how='right',
        on=['date', 'station_id'],
        suffixes=('_predicted', '_actual'),
    )

    merged.delta_stage_max_predicted.fillna(0, inplace=True)
    merged['error'] = merged.apply(rowwise_nse, axis=1)
    merged.dropna(inplace=True)
    if len(merged) != len(actual_df):
        return 'length of predicted df does not match actual df'
    else:
        score = np.divide(merged['error'].sum(), len(merged))
        return score