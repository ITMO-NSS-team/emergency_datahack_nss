import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def two_axis_plot(df, feature='snow_coverage_station_amplitude'):
    # Функция для отрисовки
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Дата')
    ax1.set_ylabel('Максимальное значение уровня, см')
    ax1.plot(df['date'], df['mean_forecast'], c='blue')
    ax1.tick_params(axis='y')
    plt.grid(c='#DCDCDC')

    ax2 = ax1.twinx()
    ax2.plot(df['date'], df[feature], c='orange')
    ax2.tick_params(axis='y')
    ax2.set_ylabel(feature)
    plt.title(f'Идентефикатор гидрологического поста - {i}')
    plt.show()


# id станций: 3019, 3027, 3028, 3029, 3030, 3035, 3041, 3045, 3230, 3050
# stage_max_amplitude, stage_max_mean, snow_coverage_station_amplitude,
# snow_height_mean, snow_height_amplitude, water_hazard_sum
path_with_files = '../../data/multi_target'
stations = [3019]
for i in stations:
    file_name = ''.join(('multi_', str(i), '.csv'))
    df = pd.read_csv(os.path.join(path_with_files, file_name),
                     parse_dates=['date'])

    # Считаем среднее значение уровня на будущие 7 дней (аналог скользящего среднего)
    df['mean_forecast'] = df['1_day'] + df['2_day'] + df['3_day'] + \
                          df['4_day'] + df['5_day'] + df['6_day'] + df['7_day']
    df['mean_forecast'] = df['mean_forecast']/7

    # Сумма рангов кодов режимной группы за определённый период
    two_axis_plot(df, 'water_hazard_sum')
    two_axis_plot(df, 'snow_height_amplitude')
    two_axis_plot(df, 'stage_max_amplitude')


