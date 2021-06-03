import pandas as pd

from sklearn.metrics import mean_absolute_error
from fedot.core.chains.chain_ts_wrappers import in_sample_ts_forecast

from models.ts_forecasting.ts_forecasting_algs import *
# Imports for creating plots
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 7

################################################################################
#     Ниже приведен алгоритм валидации прогнозирования временного ряда на      #
#                        основе in-sample прогнозироования                     #
################################################################################
# Что такое in-sample прогнозирование - https://habr.com/ru/post/559796/
# Валидация проводится не на "пологих" участках, а в период половодья


def validation(chain, predict_input, dates, forecast_length, validation_blocks,
               source_time_series):
    """ Function for validation time series forecasts on several blocks

    :param chain: fitted Chain object
    :param predict_input: InputData for prediction
    :param dates: series with dates
    :param forecast_length: forecast length
    :param validation_blocks: amount of blocks for validation
    :param source_time_series: array with time series
    """
    dates = dates.reset_index()

    # Make in-sample prediction
    horizon = 7 * validation_blocks
    predicted_values = in_sample_ts_forecast(chain=chain,
                                             input_data=predict_input,
                                             horizon=horizon)

    actual_values = np.ravel(source_time_series[-horizon:])
    pre_history = np.ravel(source_time_series[:-horizon])
    mse_metric = mean_squared_error(actual_values, predicted_values, squared=False)
    mae_metric = mean_absolute_error(actual_values, predicted_values)

    print(f'MAE - {mae_metric:.2f}')
    print(f'RMSE - {mse_metric:.2f}\n')

    # Plot time series forecasted
    plt.plot(dates, source_time_series, c='green')
    start_date = dates.iloc[len(pre_history)]
    end_date = dates.iloc[-1]
    forecast_date_range = pd.date_range(start_date['date'], end_date['date'])
    plt.plot(forecast_date_range, predicted_values, c='blue', label='Forecast')

    i = len(pre_history)
    for _ in range(0, validation_blocks):
        deviation = np.std(predicted_values)
        plt.plot([dates.iloc[i]['date'], dates.iloc[i]['date']],
                 [min(actual_values) - deviation, max(actual_values) + deviation],
                 c='black', linewidth=1)
        i += forecast_length

    plt.legend(fontsize=15)
    start_view_point = len(source_time_series) - horizon - 360
    plt.xlim(dates.iloc[start_view_point]['date'],
             dates.iloc[-1]['date'])
    plt.xlabel('Дата', fontsize=15)
    plt.ylabel('Максимальное значение уровня, см', fontsize=15)
    plt.show()


def run_validation(time_series, dates, tune_chain=False):
    # We will use 3 blocks for validation
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=7))

    # The data on which we will perform validation, the model should not have been
    # used during training
    validation_blocks = 3
    horizon = 7 * validation_blocks

    # Divide into train and test
    train_part = time_series[:-horizon]

    # InputData for train
    train_input = InputData(idx=range(0, len(train_part)),
                            features=train_part,
                            target=train_part,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # InputData for validation
    validation_input = InputData(idx=range(0, len(time_series)),
                                 features=time_series,
                                 target=time_series,
                                 task=task,
                                 data_type=DataTypesEnum.ts)

    # Обучаем цепочку
    chain = get_complex_chain('knnreg')
    chain.fit(train_input)

    if tune_chain:
        chain_tuner = ChainTuner(chain=chain, task=task, iterations=20)
        chain = chain_tuner.tune_chain(input_data=train_input,
                                       loss_function=mean_squared_error,
                                       loss_params={'squared': False})
        print('\nChain parameters after tuning')
        for node in chain.nodes:
            print(f' Operation {node.operation}, - {node.custom_params}')

    # Perform validation
    validation(chain, validation_input, dates,
               forecast_length=7,
               validation_blocks=validation_blocks,
               source_time_series=time_series)


# id станций: 3019, 3027, 3028, 3029, 3030, 3035, 3041, 3045, 3230, 3050
if __name__ == '__main__':
    # Пример формирования прогноза для 4го блока
    df = pd.read_csv('../../data/4rd_checkpoint/no_gaps_train.csv', parse_dates=['date'])
    df_submit = pd.read_csv('../../submissions/sample_submissions/sample_sub_4.csv', parse_dates=['date'])

    # Для каждого гидрологического поста строится своя прогнозная модель по временному ряду
    for station_id in [3041]:
        print(f'Предсказание формируется для станции {station_id}')
        # Данные остаются только для одной станции
        station_df = df[df['station_id'] == station_id]
        df_submit_station = df_submit[df_submit['station_id'] == station_id]

        for i in range(0, len(df_submit_station), 7):
            row = df_submit_station.iloc[i]
            current_date_for_pred = row['date']

            # Обрезаем все значения, которые идут после даты прогноза
            station_train = station_df[station_df['date'] < current_date_for_pred]
            # Если в архиве всё таки остались пропуски - убираем их
            station_train['stage_max'] = station_train[['stage_max']].interpolate(method='linear')['stage_max']
            time_series = np.ravel(np.array(station_train['stage_max']))
            # Смещаемся на 330 суток назад и смотрим как хорошо данный участок
            # будет повторять прогноз
            time_series = time_series[:-330]
            cutted = station_train['date'].head(len(station_train)-330)

            # Для сокращения времени обучения всегда берем только последние 2000
            # значений временного ряда для тренировки
            if len(time_series) > 2000:
                time_series = time_series[-2000:]
                cutted = cutted.tail(2000)
            run_validation(time_series, cutted, tune_chain=True)
