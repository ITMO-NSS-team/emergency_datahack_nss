import pandas as pd
import numpy as np
from collections import Counter

"""НЕ ЗАБУДЬТЕ:
- отсортировать df по дате.
- перевести значения колонки в int or float.
"""


def occ(x: list):
    c = Counter(x)
    c = dict(sorted(c.items(), key=lambda item: item[1]))
    res = list(c.keys())[-1]

    return res


def days_agg(dataframe: pd.DataFrame, column_name: str, agg_func: str, days: int):
    """Агрегирует данные в выбранной колонке за заданное колчество дней. По сути юзер-френдли 'скользящее окно'.

    params agg_func: модет принимать значение ['mean', 'std', 'sum', 'amplitude', 'occ'].
        - amplitude: max(value) - min(value)
        - occ: максимальное встречающееся значение
            [2,3,4,2,2,2,3]: {2: 4, 3: 2, 4: 1} => вернет 2, максимально встречающееся значение
    params days: ширина окна
    params column_name: что агрегировать
    params dataframe: dataframe

    return: ...
    """

    d = {
        'mean': np.mean,
        'sum': np.sum,
        'std': np.std,
        'amplitude': lambda x: np.max(x) - np.min(x),
        'occ': occ
    }

    agg_f = d[agg_func]
    values = np.array(dataframe[column_name])
    result = []

    for i in range(0, len(values) - days):
        result.append(agg_f(values[i: i + days]))

    return agg_f(result)
