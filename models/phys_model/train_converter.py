import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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


def get_meteo_df():
    """ Функция возвращяет интерполированные метеопараметры в узле 3045 """
    meteo_snow = pd.read_csv('../../data/meteo_data/no_gap_1day/no_gap_meteo_1day_int_3045.csv',
                             parse_dates=['date'])
    # Метеопараметры: давление и направление ветра на станции 3045
    meteo_press = pd.read_csv('../../data/meteo_data/no_gap_3hour/no_gap_meteo_3hour_int_3045_press.csv',
                              parse_dates=['date'])
    meteo_wind = pd.read_csv('../../data/meteo_data/no_gap_3hour/no_gap_meteo_3hour_int_3045_wind.csv',
                             parse_dates=['date'])
    # Объединяем датафреймы по id станции
    meteo_integrated = pd.merge(meteo_snow, meteo_press, on=['station_id', 'date'])
    meteo_integrated = pd.merge(meteo_integrated, meteo_wind, on=['station_id', 'date'])

    return meteo_integrated


if __name__ == '__main__':
    # Получаем датафрейм с метеопараметрами
    df_meteo = get_meteo_df()

    df_levels = pd.read_csv('../../data/4rd_checkpoint/no_gaps_train.csv', parse_dates=['date'])
    df_3042 = df_levels[df_levels['station_id'] == 3042]
    df_3045 = df_levels[df_levels['station_id'] == 3045]

    # Тренируем модель переводить расходы на станции 3042 в уровни на посту 3045
    df_levels = pd.merge(df_3045, df_3042, on='date', suffixes=['_3045', '_3042'])
    df_levels = df_levels[['date', 'stage_max_3045', 'discharge_3042', 'month_3045']]
    df_levels = df_levels.dropna()

    # Для избежания data leak'а натренируем модель только на первых 2000 значениях
    # df_levels = df_levels.head(2000)

    non_linear_m = RandomForestRegressor()
    x_train = np.array(df_levels[['discharge_3042', 'month_3045']])
    y_train = np.array(df_levels['stage_max_3045'])

    # Обучаем модель
    non_linear_m.fit(x_train, y_train)

    print(x_train)
    print(y_train)

    # Сохраняем модель
    filename = 'discharge_3042_into_stage_3045.pkl'
    with open(filename, 'wb') as fid:
        pickle.dump(non_linear_m, fid)
