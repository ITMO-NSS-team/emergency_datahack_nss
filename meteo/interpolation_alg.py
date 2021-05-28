import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from scipy import optimize
from copy import copy

from matplotlib import pyplot as plt
import datetime


def apply_on_dataset(meteo_df, stations_df, features_to_move, n_neighbors=2,
                     knn_model=KNeighborsRegressor):
    """ Применяет функции к нужным датафремам

    :param meteo_df: датафрейм с метеоданными
    :param stations_df: датафрейм с данными уровней
    :param features_to_move: признаки, которые нужно перемещать с метеостанций в
    "гидропосты"
    :param n_neighbors: количество соседей, по которым интерполируются значения
    """

    # Файл с координатами гидропостов
    hydro_coord = pd.read_csv('../first_data/track_2_package/hydro_coord.csv')
    # Файл с координатами метеостанций
    meteo_coord = pd.read_csv('../first_data/track_2_package/meteo_coord.csv')
    meteo_coord['station_id_nd'] = meteo_coord['station_id']

    new_df = []
    for station_id in stations_df['station_id'].unique():
        print(f'Обрабатывается гидрологическая станция {station_id}')
        # Оставляем данные только для одного гидропоста
        station_df_local = stations_df[stations_df['station_id'] == station_id]

        # Объединяем датафремы по дате
        merged = pd.merge(station_df_local, meteo_df, on='date', suffixes=['_st', '_nd'])

        # Получаем координаты гидропоста
        df_local_coords = hydro_coord[hydro_coord['station_id'] == station_id]
        x_test = df_local_coords[['lat', 'lon', 'z_null']]

        new_f_values = []
        # Для каждого признака в датафрейме c метеопараметрами
        for feature in features_to_move:
            print(f'Обрабатывается признак {feature}')
            feature = ''.join((feature, '_st'))

            # Для каждоого момента времени для гидропоста
            interpolated_values = []
            for row_id in range(0, len(station_df_local)):
                row = station_df_local.iloc[row_id]
                current_date = row['date']

                # Получаем объединенные данные для выбранного срока - один день
                merged_current = merged[merged['date'] == current_date]
                print(current_date)
                print(merged)
                print(np.array(merged_current))
                # Добавляем координаты в данные для метеостанций
                new_merged = pd.merge(merged_current, meteo_coord, on='station_id_nd')
                new_merged = new_merged.reset_index()
                print(np.array(new_merged))
                try:
                    knn = knn_model(n_neighbors)
                    target = np.array(new_merged[feature])
                    knn.fit(np.array(new_merged[['lat', 'lon', 'z']]),
                            target)
                    interpolated_v = knn.predict(x_test)[0]
                except Exception:
                    interpolated_v = None

                interpolated_values.append(interpolated_v)
            new_f_values.append(interpolated_values)
        new_f_values = np.array(new_f_values)
        print(new_f_values)


meteo_df = pd.read_csv('../first_data/track_2_package/meteo_1_sample.csv')
meteo_df['date'] = pd.to_datetime(meteo_df['date'])

stations_df = pd.read_csv('../first_data/track_2_package/train_sample.csv')
stations_df['date'] = pd.to_datetime(stations_df['date'])

apply_on_dataset(meteo_df=meteo_df,
                 stations_df=stations_df,
                 features_to_move=['day'],
                 knn_model=KNeighborsRegressor)
