import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.tuning.unified import ChainTuner
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

warnings.filterwarnings('ignore')
np.random.seed(2020)


def make_forecast(chain, train_input, predict_input, task):
    """
    Function for predicting values in a time series

    :param chain: TsForecastingChain object
    :param train_input: InputData for fit
    :param predict_input: InputData for predict
    :param task: Ts_forecasting task

    :return predicted_values: numpy array, forecast of model
    """

    # Fit it
    start_time = timeit.default_timer()
    chain.fit_from_scratch(train_input)
    amount_of_seconds = timeit.default_timer() - start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train chain\n')

    # Predict
    predicted_values = chain.predict(predict_input)
    old_predicted_values = predicted_values.predict

    chain_tuner = ChainTuner(chain=chain, task=task,
                             iterations=10)
    chain = chain_tuner.tune_chain(input_data=train_input,
                                   loss_function=mean_squared_error,
                                   loss_params={'squared': False})
    chain.save()
    print('\nChain parameters after tuning')
    for node in chain.nodes:
        print(f' Operation {node.operation}, - {node.custom_params}')

    # Predict
    predicted_values = chain.predict(predict_input)
    new_predicted_values = predicted_values.predict

    return old_predicted_values, new_predicted_values, chain


def get_complex_chain():
    """
    Chain looking like this
    smoothing - lagged - ridge \
                                \
                                 ridge -> final forecast
                                /
                lagged - ridge /
    """

    # First level
    node_smoothing = PrimaryNode('smoothing')

    # Second level
    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_smoothing])
    node_lagged_2 = PrimaryNode('lagged')

    # Third level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    # Fourth level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    chain = Chain(node_final)

    return chain


def get_ar_chain():
    """
    Function return chain with AR model
    """

    node_ar = PrimaryNode('ar')
    chain = Chain(node_ar)

    return chain


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


def run_experiment_with_tuning(time_series, with_ar_chain=False, len_forecast=250):
    """ Function with example how time series forecasting can be made

    :param time_series: time series for prediction
    :param with_ar_chain: is it needed to use chain with AR model or not
    :param len_forecast: forecast length
    """

    # Let's divide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_data,
                                                          train_data_target=train_data,
                                                          test_data_features=train_data)

    # Get chain with several models and with arima chain
    if with_ar_chain:
        chain = get_ar_chain()
    else:
        chain = get_complex_chain()

    old_predicted, new_predicted, chain = make_forecast(chain, train_input,
                                                        predict_input, task)

    old_predicted = np.ravel(np.array(old_predicted))
    new_predicted = np.ravel(np.array(new_predicted))
    test_data = np.ravel(test_data)

    mse_before = mean_squared_error(test_data, old_predicted, squared=False)
    mae_before = mean_absolute_error(test_data, old_predicted)
    print(f'RMSE before tuning - {mse_before:.4f}')
    print(f'MAE before tuning - {mae_before:.4f}\n')

    mse_after = mean_squared_error(test_data, new_predicted, squared=False)
    mae_after = mean_absolute_error(test_data, new_predicted)
    print(f'RMSE after tuning - {mse_after:.4f}')
    print(f'MAE after tuning - {mae_after:.4f}\n')

    # Готовим временной ряд
    ts_train_input, ts_predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                                train_data_features=time_series,
                                                                train_data_target=time_series,
                                                                test_data_features=time_series)

    # New_data
    new_predicted = chain.predict(ts_predict_input)
    new_predicted = np.ravel(np.array(new_predicted.predict))
    old_predicted = chain.predict(ts_predict_input)
    old_predicted = np.ravel(np.array(old_predicted.predict))
    return old_predicted, new_predicted, mse_before, mae_before, mse_after, mae_after

# 1_st_period = [21.04.1993-27.04.1993]
# 2_st_period = [21.04.1997-27.04.1997]
# 3_st_period = [21.04.2001-27.04.2001]
# 4_st_period = [21.04.2012-27.04.2012]

#date_gap = ['1993-04-21','1997-04-21','2001-04-21','2012-04-21']

if __name__ == '__main__':
    df = pd.read_csv('../data/3rd_checkpoint/no_gaps_train.csv')
    df_submit = pd.read_csv(r'C:\Users\user\Desktop\Репозитории\emergency_datahack_nss\submissions\sample_submissions\sample_sub_3.csv')
    predictions_for_post_old = []
    predictions_for_post_new = []
    mae_for_post_after = []
    mae_for_post_before = []
    mse_for_post_after = []
    mse_for_post_before = []
    num_of_post = []
    for post in df['station_id'].unique():
        try:
            tmp = df[df['station_id'] == post]
            tmp = tmp.sort_values(by=['date'])
            date_gap = df_submit[df_submit['station_id'] == post]['date'].values[::7]
            for date in date_gap:
                tmp = tmp[tmp['date'] < date]
                time_series = np.array(tmp['stage_avg'])
                old_predicted, new_predicted, mse_before, mae_before, mse_after, mae_after = \
                    run_experiment_with_tuning(time_series,
                                               with_ar_chain=False,
                                               len_forecast=7)

                num_of_post.append([date, post, old_predicted, new_predicted])
                predictions_for_post_old.append(old_predicted)
                predictions_for_post_new.append(new_predicted)
                mae_for_post_after.append(mae_after)
                mae_for_post_before.append(mae_before)
                mse_for_post_after.append(mse_before)
                mse_for_post_before.append(mse_after)

                print('Finish ' + str(post))
        except Exception:
            print('Error ' + str(post))


        with open('../submissions/predictions/all_preds', 'wb') as fp:
            pickle.dump(num_of_post, fp)

        with open('../submissions/predictions/mse_after', 'wb') as fp:
            pickle.dump(mse_for_post_after, fp)

        with open('../submissions/predictions/mse_before', 'wb') as fp:
            pickle.dump(mse_for_post_before, fp)

        with open('preds_before', 'wb') as fp:
            pickle.dump(predictions_for_post_old, fp)

        with open('preds_after', 'wb') as fp:
            pickle.dump(predictions_for_post_new, fp)


    var = 1