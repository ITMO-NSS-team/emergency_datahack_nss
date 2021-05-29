import pandas as pd
import numpy as np

from datetime import date
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 18, 7


def prepare_datetime_column(dataframe, year_col: str, month_col: str,
                            day_col: str = None, hour_col: str = None):
    datetime_col = []
    for row_id in range(0, len(dataframe)):
        row = dataframe.iloc[row_id]
        if day_col is None:
            string_datetime = ''.join((str(row[year_col]), '.', str(row[month_col])))
        elif hour_col is None:

            year = int(row[year_col])
            month = int(row[month_col])
            days = int(row[day_col])
            days_m = (date(year, month, 1) - date(year, 1, 1)).days

            string_datetime = ''.join((str(row[year_col]), '.',
                                       str(row[month_col]), '.',
                                       str(days - days_m)))
        else:
            year = int(row[year_col])
            month = int(row[month_col])
            days = int(row[day_col])
            days_m = (date(year, month, 1) - date(year, 1, 1)).days

            string_datetime = ''.join((str(row[year_col]), '.',
                                       str(row[month_col]), '.',
                                       str(days - days_m), '.',
                                       str(row[hour_col])))
        datetime_col.append(string_datetime)

    df = pd.DataFrame({'Date': datetime_col})
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d.%H')
    return df['Date']


def aggregate_speed_by_wind(dataframe, wind_col, speed_col):
    """ Функция расчитывает средние значения скорости ветра для всех направлений
    за выбранный период. Расчет производится по всему датафрейму

    :param dataframe: pandas Dataframe
    :param wind_col: название колонки с данными о направлении ветра
    :param speed_col: название колонки с данными о скорости ветра
    """

    dataframe = dataframe.groupby([wind_col]).agg({speed_col: 'mean'})
    dataframe = dataframe.reset_index()

    return dataframe


def avg_direction_and_speed(dataframe, wind_col, speed_col):
    avg_direction = dataframe[wind_col].mean()

    # Сопоставление направления ветра со средней скоростью
    speed_df = aggregate_speed_by_wind(df, wind_col, speed_col)

    # Вставляем среднее значение в датасет
    df2 = pd.DataFrame({wind_col: [avg_direction], speed_col: [np.nan]})
    speed_df = speed_df.append(df2, ignore_index=True)
    speed_df.sort_values(wind_col, inplace=True)

    # Интерполируем значения
    speed_df[speed_col] = speed_df[[speed_col]].interpolate(method='linear')[speed_col]

    # Оставляем только строку, где есть среднее значение
    calculated_row = speed_df[speed_df[wind_col] == avg_direction]

    return calculated_row


if __name__ == '__main__':
    df = pd.read_csv('track_2/extra_sample/forecast_meteo_3hours_sample.csv')
    df['date_local'] = pd.to_datetime(df['date_local'])

    print(avg_direction_and_speed(df, wind_col='wind_direction',
                                  speed_col='wind_speed_aver'))