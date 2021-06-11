import os
import warnings

import numpy as np
import pandas as pd
from models.multitarget.fedot_algs import fedot_fit_predict
from models.calculate_levels import convert_max_into_delta
warnings.filterwarnings('ignore')


# id станций: 3019, 3027, 3028, 3029, 3030, 3035, 3041, 3045, 3230, 3050
if __name__ == '__main__':
    path_with_files = '../../data/multi_target'
    df_submit = pd.read_csv('../../submissions/sample_submissions/sample_sub_4.csv', parse_dates=['date'])

    for station_id in [3050]:
        print(f'\nПредсказание формируется для станции {station_id}')

        # Read file with multi-target table for current station
        file_name = ''.join(('multi_', str(station_id), '.csv'))
        df_multi = pd.read_csv(os.path.join(path_with_files, file_name),
                               parse_dates=['date'])

        df_submit_station = df_submit[df_submit['station_id'] == station_id]
        station_forecasts = []
        for i in range(0, len(df_submit_station), 7):
            row = df_submit_station.iloc[i]
            current_date_for_pred = row['date']

            # Get train data from multi-target tables
            station_train = df_multi[df_multi['date'] < current_date_for_pred]
            station_predict_features = df_multi[df_multi['date'] == current_date_for_pred]

            # For train always use 2000 last rows
            if len(station_train) > 2000:
                station_train = station_train.tail(2000)

            # Get current value of stage_max
            current_level = np.array(station_train['1_day'])[-2]

            # Train FEDOT model and make forecast
            predict = fedot_fit_predict(station_train, station_predict_features,
                                        num_of_generations=25)

            # Функция перерасчета предсказанных значений stage_max в delta_stage_max
            deltas = convert_max_into_delta(current_level, predict)
            station_forecasts.extend(deltas)

        df_submit_station['delta_stage_max'] = station_forecasts
        path_for_save = '../../submissions/submission_data/reg_model_2'
        file_name = ''.join(('model_2_station_', str(station_id), '_.csv'))

        df_submit_station.to_csv(os.path.join(path_for_save, file_name), index=False)
