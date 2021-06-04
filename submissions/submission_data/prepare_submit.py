import os
import pandas as pd
import numpy as np


def convert_into_submit(path: str, save_name: str = 'result.csv'):
    """ Функция переводит сформированные файлы по гидрологическим постам в один файл """
    sample_sub = pd.read_csv('../sample_submissions/sample_sub_4.csv', parse_dates=['date'])

    files = os.listdir(path)
    for i, file in enumerate(files):
        if file.startswith('model'):
            df = pd.read_csv(os.path.join(path, file), parse_dates=['date'])

            # Объединяем датафреймы
            if i == 0:
                final_df = df
            else:
                frames = [final_df, df]
                final_df = pd.concat(frames)

    if len(sample_sub) != len(final_df):
        raise ValueError('Dataframes have different sizes')

    # Генерируем новый столбец в датафреймах
    sample_sub["concat"] = sample_sub["station_id"].astype(str) + sample_sub["date"].astype(str)
    final_df["concat"] = final_df["station_id"].astype(str) + final_df["date"].astype(str)

    merged_df = pd.merge(sample_sub, final_df, on='concat', suffixes=['_sample', '_preds'])
    # Формироуем новый датафрейм
    result_df = pd.DataFrame({'year': merged_df['year_sample'],
                              'station_id': merged_df['station_id_sample'],
                              'month': merged_df['month_sample'],
                              'day': merged_df['day_sample'],
                              'date': merged_df['date_sample'],
                              'delta_stage_max': merged_df['delta_stage_max_preds']})

    result_df.to_csv(os.path.join(path, save_name), index=False)


# Для модели прогнозирования временных рядов
convert_into_submit(path='ts_model_1')
