import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from scipy import optimize
from copy import copy

from matplotlib import pyplot as plt
import datetime

import warnings
warnings.filterwarnings('ignore')


def apply_on_dataset(meteo_df: pd.DataFrame, stations_df: pd.DataFrame,
                     features_to_move: list, stations_ids: list,
                     n_neighbors: int = 2, knn_model=KNeighborsRegressor, save_path='file.csv',
                     vis_station_stage=False):
    """ Применяет функции к нужным датафремам

    :param meteo_df: датафрейм с метеоданными
    :param stations_df: датафрейм с данными уровней
    :param features_to_move: признаки, которые нужно перемещать с метеостанций в
    "гидропосты"
    :param stations_ids: список идентефикаторов станций
    :param n_neighbors: количество соседей, по которым интерполируются значения
    :param knn_model: алгоритм для расчета значений
    :param save_path: куда требуется сохранить интерполированные данные
    :param vis_station_stage: требуется ли отрисовывать графики уровней
    """

    # Файл с координатами гидропостов
    hydro_coord = pd.read_csv('../../first_data/track_2_package/hydro_coord.csv')
    # Файл с координатами метеостанций
    meteo_coord = pd.read_csv('../../first_data/track_2_package/meteo_coord.csv')
    meteo_coord['station_id_meteo'] = meteo_coord['station_id']

    dates = []
    for j, station_id in enumerate(stations_ids):
        print(f'Обрабатывается гидрологическая станция {station_id}')
        # Оставляем данные только для одного гидропоста
        station_df_local = stations_df[stations_df['station_id'] == station_id]

        # Объединяем датафремы по дате
        # Одному гидрпросту в один момент времени сопоставлены несколько метеостанций
        merged = pd.merge(station_df_local, meteo_df, on='date', suffixes=['_hydro', '_meteo'])

        # Получаем координаты гидропоста
        df_local_coords = hydro_coord[hydro_coord['station_id'] == station_id]
        x_test = np.array(df_local_coords[['lat', 'lon']])

        # Для каждого признака в датафрейме c метеопараметрами
        for index, feature in enumerate(features_to_move):
            print(f'Обрабатывается признак {feature}')

            # Для каждоого момента времени для гидропоста
            interpolated_values = []
            start_date = min(station_df_local['date'])
            end_date = max(station_df_local['date'])
            all_dates = pd.date_range(start_date, end_date)

            if vis_station_stage:
                plt.plot(station_df_local['date'], station_df_local['stage_max_hydro'])
                plt.xlabel('Дата', fontsize=15)
                plt.ylabel('Максимальное значение уровня, см', fontsize=15)
                plt.title(f'Hydro station id - {station_id}')
                plt.show()

            for current_date in all_dates:
                # Получаем объединенные данные для выбранного срока - один день
                merged_current = merged[merged['date'] == current_date]
                # Добавляем координаты в данные для метеостанций
                new_merged = pd.merge(merged_current, meteo_coord, on='station_id_meteo')
                new_merged = new_merged.reset_index()

                try:
                    # По координатам и высоте прогнозируем значение в гидропосте
                    dataset = new_merged[['lat', 'lon', feature]]
                    # Убираем экстремально высокие значения - это пропуски
                    if feature == 'snow_coverage_station':
                        dataset[dataset[feature] > 50] = np.nan
                    else:
                        dataset[dataset[feature] > 9000] = np.nan
                    dataset = dataset.dropna()

                    knn = knn_model(n_neighbors)
                    target = np.array(dataset[feature])
                    knn.fit(np.array(dataset[['lat', 'lon']]), target)
                    interpolated_v = knn.predict(x_test)[0]
                except Exception:
                    # Значит значения содержат пропуски
                    interpolated_v = None

                interpolated_values.append(interpolated_v)

            if index == 0:
                # Добавляем даты
                dates.extend(all_dates)
                new_f_values = np.array(interpolated_values).reshape((-1, 1))
            else:
                int_column = np.array(interpolated_values).reshape((-1, 1))
                # Присоединяем дополнительный признак справа
                new_f_values = np.hstack((new_f_values, int_column))

        if j == 0:
            # Датафрем с интерполированными значениями
            new_station_info = pd.DataFrame(new_f_values, columns=features_to_move)
            new_station_info['station_id'] = [station_id] * len(new_station_info)
            new_station_info.to_csv('file.csv')
        else:
            new_dataframe = pd.DataFrame(new_f_values, columns=features_to_move)
            new_dataframe['station_id'] = [station_id] * len(new_dataframe)

            # Добавляем уже к сформированному датафрему
            frames = [new_station_info, new_dataframe]
            new_station_info = pd.concat(frames)

    new_station_info['date'] = dates
    new_station_info.to_csv(save_path, index=False)

# Гидрологические посты для которых требуются данные
# 3019, 3027, 3028, 3029, 3030, 3035, 3041, 3045, 3230, 3050

# Параметры 3h - 'air_temperature', 'relative_humidity', 'pressure' - press
# 'wind_direction', 'wind_speed_aver', 'precipitation' - wind

# Параметры 1day - 'snow_coverage_station', 'snow_height'

meteo_df = pd.read_csv('../../first_data/track_2_package/meteo_1day.csv')
meteo_df['date'] = pd.to_datetime(meteo_df['date'])

stations_df = pd.read_csv('../../first_data/track_2_package/train.csv')
stations_df['date'] = pd.to_datetime(stations_df['date'])

apply_on_dataset(meteo_df=meteo_df,
                 stations_df=stations_df,
                 features_to_move=['snow_coverage_station', 'snow_height'],
                 knn_model=KNeighborsRegressor,
                 n_neighbors=3,
                 save_path='gap_meteo_3036.csv',
                 stations_ids=[3036],
                 vis_station_stage=True)
