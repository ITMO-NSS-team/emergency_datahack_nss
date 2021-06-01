"""НЕ ЗАБУДЬТЕ:
- отсортировать df по дате.
- перевести значения колонки в int or float.
"""

import numpy as np
import pandas as pd
from collections import Counter
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


def occ(x: list):
    c = Counter(x)
    c = dict(sorted(c.items(), key=lambda item: item[1]))
    res = list(c.keys())[-1]

    return res


def days_agg(dataframe: pd.DataFrame, column_name: str, agg_func: str, days: int):
    """Агрегирует данные в выбранной колонке за заданное колчество дней. По сути юзер-френдли 'скользящее окно'.

    params agg_func: модет принимать значение ['mean', 'std', 'sum', 'amplitude', 'occ'].
        - amplitude: max(value) - min(value)
        - occ: максимальное встречающееся значение
            [2,3,4,2,2,2,3]: {2: 4, 3: 2, 4: 1} => вернет 2, максимально встречающееся значение
    params days: ширина окна
    params column_name: что агрегировать
    params dataframe: dataframe

    return: ...
    """

    d = {
        'mean': np.mean,
        'sum': np.sum,
        'std': np.std,
        'amplitude': lambda x: np.max(x) - np.min(x),
        'occ': occ
    }

    agg_f = d[agg_func]
    values = np.array(dataframe[column_name])
    result = []

    for i in range(0, len(values)):
        i_ = i - days
        if i_ < 0:
            i_ = 0

        result.append(agg_f(values[i_:i + 1]))

    return result


def ts_to_table(idx, time_series, window_size):
    """ Method convert time series to lagged form.
    :param idx: the indices of the time series to convert
    :param time_series: source time series
    :param window_size: size of sliding window, which defines lag
    :return updated_idx: clipped indices of time series
    :return features_columns: lagged time series feature table
    """

    # Convert data to lagged form
    lagged_dataframe = pd.DataFrame({'t_id': time_series})
    vals = lagged_dataframe['t_id']
    for i in range(1, window_size + 1):
        frames = [lagged_dataframe, vals.shift(i)]
        lagged_dataframe = pd.concat(frames, axis=1)

    # Remove incomplete rows
    lagged_dataframe.dropna(inplace=True)

    transformed = np.array(lagged_dataframe)

    # Generate dataset with features
    features_columns = transformed[:, 1:]
    features_columns = np.fliplr(features_columns)

    return idx, features_columns


def convert_water_codes(value):
    water_codes = pd.read_csv('../../data/meteo_data/ref_code_hazard.csv')
    values = list(map(int, map(lambda x: x.strip(), value.split(','))))
    res = 0

    for val in values:
        res += water_codes[water_codes['water_code'] == val].reset_index(drop=True).iloc[0][1]

    return res


def feature_aggregation(new_df):
    columns = new_df.columns
    columns_drop = []

    if 'discharge' in columns:
        new_df['discharge_mean'] = days_agg(new_df, 'discharge', 'mean', 4)
        columns_drop.append('discharge')
    if 'stage_avg' in columns:
        new_df['stage_avg_amplitude'] = days_agg(new_df, 'stage_avg', 'amplitude', 7)
        new_df['stage_avg_mean'] = days_agg(new_df, 'stage_avg', 'mean', 4)
        columns_drop.append('stage_avg')
    if 'snow_coverage_station' in columns:
        new_df['snow_coverage_station_amplitude'] = days_agg(new_df, 'snow_coverage_station', 'amplitude', 7)
        columns_drop.append('snow_coverage_station')
    if 'snow_height' in columns:
        new_df['snow_height_mean'] = days_agg(new_df, 'snow_height', 'mean', 4)
        new_df['snow_height_amplitude'] = days_agg(new_df, 'snow_height', 'amplitude', 7)
        columns_drop.append('snow_height')
    if 'precipitation' in columns:
        new_df['precipitation_sum'] = days_agg(new_df, 'precipitation', 'sum', 30)
        columns_drop.append('precipitation')
    if 'water_hazard' in columns:
        new_df['water_hazard_sum'] = days_agg(new_df, 'water_hazard', 'sum', 2)
        columns_drop.append('water_hazard')

    new_df.drop(columns_drop, axis=1, inplace=True)

    return new_df
