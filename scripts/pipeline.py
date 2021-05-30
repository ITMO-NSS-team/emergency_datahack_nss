import pickle
import numpy as np
import pandas as pd
from collections import Counter
import warnings

#building models
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_auc_score, roc_curve

# from fedot.api.main import Fedot
import xgboost as xgb
import lightgbm as lgb

# from lightautoml.automl.base import AutoML
# from lightautoml.ml_algo.boost_lgbm import BoostLGBM
# from lightautoml.ml_algo.tuning.optuna import OptunaTuner
# from lightautoml.automl.blend import WeightedBlender
# from lightautoml.ml_algo.boost_cb import BoostCB
# from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
#
# from lightautoml.pipelines.ml.base import MLPipeline
# from lightautoml.reader.base import PandasToPandasReader
# from lightautoml.tasks import Task


warnings.simplefilter(action='ignore', category=FutureWarning)


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


def _ts_to_table(idx, time_series, window_size):
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
    values = list(map(int, map(lambda x: x.strip(), value.split(','))))
    res = 0

    for val in values:
        res += water_codes[water_codes['water_code'] == val].reset_index(drop=True).iloc[0][1]

    return res


sample_sub_1 = pd.read_csv('./submissions/sample_submissions/sample_sub_4.csv')

meteo_prep = pd.read_csv('./data/meteo_data/no_gap_meteo_3hour_int_3029_wind.csv')
meteo_prep.drop(['station_id'], inplace=True, axis=1)

meteo = pd.read_csv('./data/meteo_data/no_gap_meteo_1day_int_3019.csv')
meteo.drop(['station_id'], inplace=True, axis=1)
meteo.columns = ['date', 'snow_coverage_station', 'snow_height']

water_codes = pd.read_csv('./data/misc/ref_code_hazard.csv')


ids = [3019, 3027, 3028, 3030, 3035, 3041, 3045, 3230, 3050, 3029]

for id_ in ids:
    train = pd.read_csv(f'./data/2nd_checkpoint/sub_datasets_no_gaps/no_gaps/no_gap_train_{id_}.csv')
    train = train.set_index('date')
    train['water_hazard'] = train['water_code'].fillna('1').apply(convert_water_codes)
    train.drop(['stage_min', 'stage_max', 'temp', 'water_code', 'station_id', 'ice_thickness',
                'snow_height', 'place', 'year', 'month', 'day', 'delta_stage_max'], axis=1, inplace=True)

    columns = ['0_day', '1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']
    target_column = 'stage_avg'

    a = train[target_column]
    window_size = 8
    idx = np.arange(train.shape[0] - window_size) + 1
    idx, b = _ts_to_table(idx, a, window_size)

    train = train.iloc[:-window_size, :]

    new_df = pd.DataFrame(data=b, columns=columns, index=a.index[:-window_size])
    new_df = pd.concat([train, new_df], axis=1, join='inner')
    new_df.drop(['0_day'], axis=1, inplace=True)

    new_df = pd.merge(meteo_prep, new_df, how='inner', on=['date'])
    new_df = pd.merge(meteo, new_df, how='inner', on=['date'])
    new_df.drop(['wind_direction', 'wind_speed_aver'], axis=1, inplace=True)

    predictions = []
    columns = ['1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']

    for index in range(0, len(sample_sub_1['date']), 7):
        mini_df = new_df[new_df['date'] < sample_sub_1['date'][index]]
        mini_df = feature_aggregation(mini_df)
        mini_df.drop(['date'], inplace=True, axis=1)

        features = np.array(mini_df.drop(columns, axis=1))
        features_test = np.array(mini_df.drop(columns, axis=1).iloc[-1, :]).reshape(1, -1)
        target = np.array(mini_df[columns])

        model = MultiOutputRegressor(lgb.LGBMRegressor(random_state=42), n_jobs=-1)
        model.fit(features, target)
        pred = model.predict(features_test)
        predictions.append(pred)

    with open(f'predictions{id_}.pkl', 'wb') as f:
        pickle.dump(predictions, f)

        # composer_params = {'max_depth': 5,
        #                    'max_arity': 7,
        #                    'pop_size': 5,
        #                    'num_of_generations': 20,
        #                    'learning_time': 10}
        # model = Fedot(problem='regression', preset='light_tun', learning_time=13, composer_params=composer_params, seed=42)
        # model.fit(features=features, target=target)
        # pred = model.predict(features_test)
        # predictions.append(pred.predict)


# task = Task('reg')
# reader = PandasToPandasReader(task, cv=5, random_state=1)
#
# model1 = BoostLGBM(default_params={'learning_rate': 0.1, 'num_leaves': 128, 'seed': 1, 'num_threads': 5})
# params_tuner2 = OptunaTuner(n_trials=100, timeout=100)
# model2 = BoostLGBM(default_params={'learning_rate': 0.05, 'num_leaves': 64, 'seed': 2, 'num_threads': 5})
# gbm_0 = BoostCB()
# gbm_1 = BoostCB()
# tuner_0 = OptunaTuner(n_trials=100, timeout=100, fit_on_holdout=True)
#
#
# pipeline_lvl1 = MLPipeline([model1, (model2, params_tuner2), (gbm_0, tuner_0), gbm_1])
# reg_2 = LinearLBFGS()
# pipeline_lvl2 = MLPipeline([reg_2])
#
# predictions = []
#
#
# timer = PipelineTimer(600, mode=2)
# automl = AutoML(reader, [
#     [pipeline_lvl1],
#     [pipeline_lvl2],
# ], skip_conn=False, blender=WeightedBlender(), timer=timer)
# pred = automl.fit_predict(new_df.drop(columns, axis=1), roles={'target': columns})
# predictions.append(pred)
