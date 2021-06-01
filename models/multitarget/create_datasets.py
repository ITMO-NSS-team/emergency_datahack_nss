import os
import pandas as pd
import numpy as np

from models.multitarget.aggregation import convert_water_codes, ts_to_table

""" Ниже представлен алгоритм формирования multi-target таблци на агрегированных 
признаках для подачи их в алгоритм прогнозирования 

Способы агрегирования:
    'mean': np.mean, среднее арифметическое за выбранный период
    'sum': np.sum, сумма значений за выбранный период
    'std': np.std, стандартное отклонение за выбранный период
    'amplitude': lambda x: np.max(x) - np.min(x), амплитуда значений
    'occ': occ, наиболее часто встречающиеся значения за период
"""

if __name__ == '__main__':
    main_columns = ['0_day', '1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']
    target_column = 'stage_max'

    # Пути до файлов с метеопараметрами
    meteo_1d_path = '../../data/meteo_data/no_gap_1day'
    meteo_3h_path = '../../data/meteo_data/no_gap_3hour'

    df_submit = pd.read_csv('../../submissions/sample_submissions/sample_sub_4.csv',
                            parse_dates=['date'])
    df = pd.read_csv('../../data/4rd_checkpoint/no_gaps_train.csv',
                     parse_dates=['date'])

    # Для каждоого гидропоста производим агрегацию данных
    for station_id in df_submit['station_id'].unique():
        print(f'\nТаблица формируется для станции {station_id}')

        # Оставляем данные только для одного гидропоста
        station_df = df[df['station_id'] == station_id]
        station_df = station_df.set_index('date')
        station_df['water_hazard'] = station_df['water_code'].fillna('1').apply(convert_water_codes)
        station_df = station_df[['water_hazard', target_column]]

        station_submit = df_submit[df_submit['station_id'] == station_id]

        # Читаем данные по снежному покрову
        snow_name = ''.join(('no_gap_meteo_1day_int_', str(station_id), '.csv'))
        snow_data = pd.read_csv(os.path.join(meteo_1d_path, snow_name),
                                parse_dates=['date'])

        # Объединяем датафреймы по столбцу "дата"
        merged_df = pd.merge(station_df, snow_data, on='date')
        merged_df = merged_df.sort_values(by=['date'])

        # Готовим датафрейм
        window_size = 8
        idx = np.arange(merged_df.shape[0] - window_size) + 1
        target_arr = merged_df[target_column]
        _, trs_target = ts_to_table(idx, target_arr, window_size)

        # TODO дописать функцию генерации таблиц многомерной регрессии
