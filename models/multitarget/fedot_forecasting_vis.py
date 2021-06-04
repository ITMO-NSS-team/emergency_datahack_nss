import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from models.multitarget.fedot_algs import dataframe_into_inputs, fedot_fit
from pylab import rcParams
rcParams['figure.figsize'] = 15, 7


def multi_validation(station_train, val_blocks=3):
    horizon = val_blocks * 7
    cutted_df = station_train.head(len(station_train) - horizon)
    val_df = station_train.tail(horizon)

    train_input = dataframe_into_inputs(cutted_df)

    # Get chain after composing
    chain = fedot_fit(train_input, num_of_generations=10)

    forecasts = []
    for i in range(0, horizon, 7):
        row = val_df.iloc[i]
        row_input = dataframe_into_inputs(row)

        preds = chain.predict(row_input)
        forecast = list(np.ravel(np.array(preds.predict)))
        forecasts.extend(forecast)

    forecasts_df = pd.DataFrame({'date': val_df['date'], 'predict': forecasts})

    # Convert station_train into time-series dataframe
    station_train['stage_max'] = station_train['1_day'].shift(1)
    # Remove first row
    station_train = station_train.tail(len(station_train) - 1)
    new_forecasts_df = pd.merge(forecasts_df, station_train, on='date')

    plt.plot(station_train['date'], station_train['stage_max'], c='green', label='Actual time series')
    plt.plot(forecasts_df['date'], forecasts_df['predict'], c='blue', label='Forecast')

    i = len(cutted_df) - 1
    dates = station_train['date']
    dates = dates.reset_index()
    actual_values = np.array(new_forecasts_df['stage_max'])
    for _ in range(0, val_blocks):
        deviation = np.std(np.array(new_forecasts_df['predict']))
        plt.plot([dates.iloc[i]['date'], dates.iloc[i]['date']],
                 [min(actual_values) - deviation, max(actual_values) + deviation],
                 c='black', linewidth=1)
        i += 7

    plt.xlabel('Дата', fontsize=15)
    plt.ylabel('Максимальное значение уровня, см', fontsize=15)
    start_view_point = len(station_train) - horizon - 360
    plt.xlim(dates.iloc[start_view_point]['date'],
             dates.iloc[-1]['date'])
    plt.show()


# id станций: 3019, 3027, 3028, 3029, 3030, 3035, 3041, 3045, 3230, 3050
if __name__ == '__main__':
    path_with_files = '../../data/multi_target'
    df_submit = pd.read_csv('../../submissions/sample_submissions/sample_sub_4.csv', parse_dates=['date'])

    for station_id in [3230]:
        print(f'\nПредсказание формируется для станции {station_id}')

        # Read file with multi-target table for current station
        file_name = ''.join(('multi_', str(station_id), '.csv'))
        df_multi = pd.read_csv(os.path.join(path_with_files, file_name),
                               parse_dates=['date'])

        df_submit_station = df_submit[df_submit['station_id'] == station_id]
        for i in range(0, len(df_submit_station), 7):
            row = df_submit_station.iloc[i]
            current_date_for_pred = row['date']

            # Get train data from multi-target tables
            station_train = df_multi[df_multi['date'] < current_date_for_pred]
            station_predict_features = df_multi[df_multi['date'] == current_date_for_pred]

            # For train always use 2500 last rows
            if len(station_train) > 2500:
                station_train = station_train.tail(2500)

            # Смещаемся на 330 суток назад и смотрим как хорошо данный участок
            # будет повторять прогноз
            station_train = station_train.head(2170)

            # Валидация
            multi_validation(station_train)
