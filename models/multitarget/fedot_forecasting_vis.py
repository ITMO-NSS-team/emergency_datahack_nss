import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from models.multitarget.fedot_algs import dataframe_into_inputs


def multi_validation(station_train, val_blocks=3):
    # Create simple chain
    node_rfr = PrimaryNode('rfr')
    chain = Chain(node_rfr)

    horizon = val_blocks * 7
    cutted_df = station_train.head(len(station_train) - horizon)
    val_df = station_train.tail(horizon)

    train_input = dataframe_into_inputs(cutted_df)
    chain.fit(train_input)

    forecasts = []
    for i in range(0, horizon, 7):
        row = val_df.iloc[i]
        row_input = dataframe_into_inputs(row)

        preds = chain.predict(row_input)
        forecast = list(np.ravel(np.array(preds.predict)))
        forecasts.extend(forecast)

    # TODO добавить визуализации
    print(forecasts)


# id станций: 3019, 3027, 3028, 3029, 3030, 3035, 3041, 3045, 3230, 3050
if __name__ == '__main__':
    path_with_files = '../../data/multi_target'
    df_submit = pd.read_csv('../../submissions/sample_submissions/sample_sub_4.csv', parse_dates=['date'])

    for station_id in [3035]:
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
