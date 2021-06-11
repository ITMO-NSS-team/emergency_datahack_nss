import os
import pickle
import pandas as pd
import numpy as np
from models.calculate_levels import convert_max_into_delta
from models.phys_model.srm_model import fit_3045_phys_model, get_const_for_3045
from models.phys_model.train_converter import get_meteo_df


""" Физическая модель прогнозирования расхода вода на гидрологическом посту 3045.
На гидрологическом посту 3045 нет данных о расходах, но есть значения уровней. 

Для моделирования требуется значения с соседних станций:
    - 3036 (расход), станция, расположенная выше по течению реки
    - 3042 (расход), станция, расположенная близко к 3045. Можем принять, что 
значения расходов на станции 3042 хорошо взаимосвязаны с расходом и уровнем воды 
на моделируемой станции 3045.

Физическая модель предсказывает расходы, не уровни! Перерасчет расходов в уровни 
осуществляется при помощи ML модели (случайного леса). 
"""


def load_converter(filename='discharge_3042_into_stage_3045.pkl'):
    with open(filename, 'rb') as f:
        clf = pickle.load(f)

    return clf


def get_all_data_for_3045_forecasting():
    df_meteo = get_meteo_df()

    df_levels = pd.read_csv('../../data/4rd_checkpoint/no_gaps_train.csv', parse_dates=['date'])
    df_3042 = df_levels[df_levels['station_id'] == 3042]
    df_3036 = df_levels[df_levels['station_id'] == 3036]
    df_3045 = df_levels[df_levels['station_id'] == 3045]

    df_hydro = pd.merge(df_3042, df_3036, on='date', suffixes=['_3042', '_3036'])
    df_merge = pd.merge(df_meteo, df_hydro)
    df_merge = pd.merge(df_merge, df_3045, on='date')
    return df_merge


def convert_discharge_into_stage_max(model, forecast, months):
    """ Перерасчет расхода в уровне на основе обученной модели """
    forecast = np.array(forecast).reshape((-1, 1))
    months = np.array(months).reshape((-1, 1))

    x_test = np.hstack((forecast, months))
    # Предсказание уровня на основе расходов
    stage_max = model.predict(x_test)
    stage_max = np.ravel(stage_max)
    return stage_max


if __name__ == '__main__':
    cwd = os.getcwd()
    meteo_df = get_meteo_df()

    # Обучаем модель только на данных нужного периода: весна и лето
    start_month = 4
    end_month = 9
    mask = (meteo_df['date'].dt.month >= start_month) & (meteo_df['date'].dt.month <= end_month)
    meteo_spring = meteo_df.loc[mask]
    meteo_spring = meteo_spring.dropna()

    # Обучение физической модели
    dm = fit_3045_phys_model(meteo_spring)

    # Загружаем модель машинного обучения, которая будет конвертировать расходы в уровни
    convert_model = load_converter()

    # Загружаем все необходимые данные
    df_merge = get_all_data_for_3045_forecasting()

    df_submit = pd.read_csv('../../submissions/sample_submissions/sample_sub_4.csv', parse_dates=['date'])
    df_submit = df_submit[df_submit['station_id'] == 3045]

    # Неизменяемые параметры для станции номер 3045
    params = get_const_for_3045()
    const_area = params['area']
    section_to = params['section_to']
    lapse = params['lapse']
    h_mean = params['h_mean']
    h_st = params['h_st']

    # Пресдсказание алгоритма
    forecasts = []
    for i in range(0, len(df_submit), 7):
        row = df_submit.iloc[i]
        current_date_for_pred = row['date']

        # Получаем данные только до нужной даты, на которую даём прогноз
        local_df = df_merge[df_merge['date'] < current_date_for_pred]
        # Берём текущее значения уровня воды на станции 3045
        current_level = np.array(local_df['stage_max'])[-1]

        # Получаем данные о текущем месяце
        local_sb_df = df_submit.iloc[i:i+7]

        # Задаем преикторы в модель
        last_row = local_df.iloc[-1]
        # Рассчитываем параметр температуры
        tmp = last_row['air_temperature'] + lapse * (h_mean - h_st) * 0.01
        snow_cover = last_row['snow_coverage_station']
        rainfall = last_row['precipitation']
        disch_3042 = last_row['discharge_3042']
        disch_3036 = last_row['discharge_3036']
        start_variables = np.array([tmp, snow_cover, const_area, rainfall, disch_3042, disch_3036])
        start_variables = np.nan_to_num(start_variables)
        start_variables = tuple(start_variables)

        # Метеопарамеры
        lr_col = last_row[['snow_height_y', 'snow_coverage_station', 'air_temperature',
                           'relative_humidity', 'pressure', 'wind_direction', 'wind_speed_aver',
                           'precipitation']]
        lr_col = lr_col.fillna(value=0.0)
        start_meteodata = np.array(lr_col)
        start_meteodata = np.nan_to_num(start_meteodata)

        # Прогноз при помощи физической модели
        forecast = dm.predict_period(start_variables, start_meteodata, period=7)

        # Трансформация расходов в уровни
        stage = convert_discharge_into_stage_max(model=convert_model,
                                                 forecast=forecast,
                                                 months=local_sb_df['month'])

        # Функция перерасчета предсказанных значений stage_max в delta_stage_max
        deltas = convert_max_into_delta(current_level, stage)
        forecasts.extend(deltas)

    df_submit['delta_stage_max'] = forecasts
    path_for_save = '../../submissions/submission_data/phys_model_3'
    file_name = 'model_3_station_3045_.csv'

    df_submit.to_csv(os.path.join(path_for_save, file_name), index=False)
