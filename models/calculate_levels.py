import numpy as np


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
