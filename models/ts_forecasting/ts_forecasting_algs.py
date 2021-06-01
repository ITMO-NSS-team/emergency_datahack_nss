import timeit
import numpy as np
from sklearn.metrics import mean_squared_error
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.tuning.unified import ChainTuner
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

import warnings
warnings.filterwarnings('ignore')


def get_complex_chain(model_name: str = 'ridge'):
    """
    Chain looking like this
    smoothing - lagged - ridge  \
                                 \
                                  ridge -> final forecast
                                 /
                lagged - ridge* /
    *гиперпараметр
    """

    # First level - Слгаживание временного ряда
    node_smoothing = PrimaryNode('gaussian_filter')
    node_smoothing.custom_params = {'sigma': 3}

    # Second level
    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_smoothing])
    node_lagged_1.custom_params = {'window_size': 100}
    node_lagged_2 = PrimaryNode('lagged')
    node_lagged_2.custom_params = {'window_size': 10}

    # Third level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_model_2 = SecondaryNode(model_name, nodes_from=[node_lagged_2])

    # Fourth level - root node
    node_final = SecondaryNode('linear', nodes_from=[node_ridge_1, node_model_2])
    chain = Chain(node_final)

    return chain


def make_forecast(chain, train_input, predict_input, task, tune_chain=False,
                  save_chain=False):
    """
    Function for predicting values in a time series

    :param chain: TsForecastingChain object
    :param train_input: InputData for fit
    :param predict_input: InputData for predict
    :param task: Ts_forecasting task
    :param tune_chain: is it needed to tune chain
    :param save_chain: is it needed to serialise chain

    :return forecast: numpy array, forecast of model
    """

    # Fit it
    start_time = timeit.default_timer()
    chain.fit(train_input)

    # Требуется ли настраивать гиперпараметры в узлах цепочки
    if tune_chain:
        chain_tuner = ChainTuner(chain=chain, task=task, iterations=20)
        chain = chain_tuner.tune_chain(input_data=train_input,
                                       loss_function=mean_squared_error,
                                       loss_params={'squared': False})
        print('\nChain parameters after tuning')
        for node in chain.nodes:
            print(f' Operation {node.operation}, - {node.custom_params}')

    amount_of_seconds = timeit.default_timer() - start_time
    print(f'It takes {amount_of_seconds:.2f} seconds to train chain')

    if save_chain:
        chain.save('')

    predicted_values = chain.predict(predict_input)
    forecast = predicted_values.predict

    return forecast


def prepare_input_data(len_forecast, train_data_features, train_data_target,
                       test_data_features):
    """ Function return prepared data for fit and predict

    :param len_forecast: forecast length
    :param train_data_features: time series which can be used as predictors for train
    :param train_data_target: time series which can be used as target for train
    :param test_data_features: time series which can be used as predictors for prediction

    :return train_input: Input Data for fit
    :return predict_input: Input Data for predict
    :return task: Time series forecasting task with parameters
    """

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_data_features)),
                            features=train_data_features,
                            target=train_data_target,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_data_features)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=test_data_features,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, task


def launch_ts_forecasting_on_station(time_series, chain, tune_chain=False, save_chain=False):
    """ Функция строит модель для прогнозирования временного ряда и даёт
    прогноз на 7 элементов вперед
    """
    train_input, predict_input, task = prepare_input_data(len_forecast=7,
                                                          train_data_features=time_series,
                                                          train_data_target=time_series,
                                                          test_data_features=time_series)

    forecast = make_forecast(chain, train_input, predict_input, task,
                             tune_chain, save_chain)

    return np.ravel(forecast)


def convert_max_into_delta(observed_level, predicted_max):
    """ Функция переводит предсказанные значения уровня (stage_max) в целевую
    переменную delta_stage_max

    :param observed_level: значение уровня, которое было известно на начало прогноза
    :param predicted_max: предсказанные значения уровней на 7 дней вперед
    :return delta_levels: разница уровней
    """
    shifted = predicted_max[:-1]
    new_arr = np.hstack([np.array(observed_level), shifted])

    delta_stage_max_predicted = predicted_max - new_arr
    return delta_stage_max_predicted
