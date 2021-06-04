import os
from models.interpolation.interpolation_alg import *

################################################################################
#      Ниже приведен пример запуска алгоритма интерполяции метеопараметров     #
################################################################################
""" Гидрологические посты для которых требуются данные
id станций: 3019, 3027, 3028, 3029, 3030, 3035, 3041, 3045, 3230, 3050

Параметры 3h данные с метеостанций по срокам (каждые 3 часа)
'air_temperature', 'relative_humidity', 'pressure' - press
'wind_direction', 'wind_speed_aver', 'precipitation' - wind

Параметры 1day (ежедневные данные)
'snow_coverage_station', 'snow_height'
"""

meteo_df = pd.read_csv('../../first_data/track_2_package/meteo_1day.csv')
meteo_df['date'] = pd.to_datetime(meteo_df['date'])

stations_df = pd.read_csv('../../data/4rd_checkpoint/no_gaps_train.csv')
stations_df['date'] = pd.to_datetime(stations_df['date'])

# Для каждого гидрологического поста производим интерполяцию
for station_i in [3019, 3027, 3028, 3029, 3030, 3035, 3041, 3045, 3230, 3050]:
    save_file_name = ''.join(('gap_meteo_', str(station_i), '.csv'))
    save_path = os.path.join('../../data/meteo_data', save_file_name)
    apply_on_dataset(meteo_df=meteo_df,
                     stations_df=stations_df,
                     features_to_move=['snow_height'],
                     knn_model=KNeighborsRegressor,
                     n_neighbors=3,
                     save_path=save_path,
                     stations_ids=[station_i],
                     vis_station_stage=True)
