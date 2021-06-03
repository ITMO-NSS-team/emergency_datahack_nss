import datetime
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


def dataframe_into_inputs(dataframe):
    """ Function converts pandas DataFrame into InputData FEDOT format

    :param dataframe: pandas Dataframe with data
    """
    target_columns = ['1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']
    features_columns = ['stage_max_amplitude', 'stage_max_mean', 'snow_coverage_station_amplitude',
                        'snow_height_mean', 'snow_height_amplitude', 'water_hazard_sum']

    # Get features and targets arrays
    targets = np.array(dataframe[target_columns])
    if len(targets.shape) == 1:
        targets = targets.reshape((1, -1))
    features = np.array(dataframe[features_columns])
    if len(features.shape) == 1:
        features = features.reshape((1, -1))

    task = Task(TaskTypesEnum.regression)
    input_data = InputData(idx=np.arange(0, len(features)), features=features,
                           target=targets, task=task, data_type=DataTypesEnum.table)

    return input_data


def fedot_fit(train_data, num_of_generations):
    # Create simple chain
    node_scaling = PrimaryNode('scaling')
    node_ridge = SecondaryNode('rfr', nodes_from=[node_scaling])
    init_chain = Chain(node_ridge)

    available_operations_types = ['ridge', 'lasso', 'dtreg',
                                  'xgbreg', 'adareg', 'rfr',
                                  'linear', 'svr', 'poly_features',
                                  'scaling', 'ransac_lin_reg', 'rfe_lin_reg',
                                  'pca', 'ransac_non_lin_reg',
                                  'rfe_non_lin_reg', 'normalization']
    composer_requirements = GPComposerRequirements(
        primary=available_operations_types,
        secondary=available_operations_types, max_arity=3,
        max_depth=8, pop_size=10, num_of_generations=num_of_generations,
        crossover_prob=0.8, mutation_prob=0.8,
        max_lead_time=datetime.timedelta(minutes=5),
        allow_single_operations=False)
    mutation_types = [MutationTypesEnum.parameter_change, MutationTypesEnum.simple,
                      MutationTypesEnum.reduce]
    optimiser_parameters = GPChainOptimiserParameters(mutation_types=mutation_types)

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.MAE)
    builder = GPComposerBuilder(task=train_data.task). \
        with_optimiser_parameters(optimiser_parameters). \
        with_requirements(composer_requirements). \
        with_metrics(metric_function).with_initial_chain(init_chain)
    composer = builder.build()
    obtained_chain = composer.compose_chain(data=train_data, is_visualise=False)

    # Fit chain after composing
    obtained_chain.fit(train_data)

    return obtained_chain


def fedot_fit_predict(station_train: pd.DataFrame,
                      station_predict_features: pd.DataFrame,
                      num_of_generations: int = 10):
    """ Функция запускает композирование цепочки и после обучения даёт прогноз

    :param station_train:
    :param station_predict_features:
    :param num_of_generations:
    """

    # Wrap dataframe into output
    train_data = dataframe_into_inputs(station_train)
    test_data = dataframe_into_inputs(station_predict_features)

    obtained_chain = fedot_fit(train_data, num_of_generations)

    predicted_output = obtained_chain.predict(test_data)
    # Convert output into one dimensional array
    forecast = np.ravel(np.array(predicted_output.predict))

    return forecast




