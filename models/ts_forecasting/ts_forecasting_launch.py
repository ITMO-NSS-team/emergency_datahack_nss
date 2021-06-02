import pandas as pd
from models.ts_forecasting.ts_forecasting_algs import *
from models.calculate_levels import convert_max_into_delta


if __name__ == '__main__':
    # Пример формирования прогноза для 4го блока
    df = pd.read_csv('../../data/4rd_checkpoint/no_gaps_train.csv', parse_dates=['date'])
    df_submit = pd.read_csv('../../submissions/sample_submissions/sample_sub_4.csv', parse_dates=['date'])

    # Для каждого гидрологического поста строится своя прогнозная модель по временному ряду
    forecasts = []
    for station_id in df_submit['station_id'].unique():
        print(f'\nПредсказание формируется для станции {station_id}')
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

            # Для сокращения времени обучения всегда берем только последние 2000
            # значений временного ряда для тренировки
            if len(time_series) > 2000:
                time_series = time_series[-2000:]

            # Берем значение уровня в данный момент времени
            current_level = time_series[-1]

            # Объявляем цепочку для прогноза
            chain_model = get_complex_chain('knnreg')
            predict = launch_ts_forecasting_on_station(time_series, chain_model,
                                                       tune_chain=True, save_chain=True)

            # Функция перерасчета предсказанных значений stage_max в delta_stage_max
            deltas = convert_max_into_delta(current_level, predict)
            forecasts.extend(deltas)

    # Записываем предсказания в датафрейм
    df_submit['delta_stage_max'] = forecasts
    # Сохраняем в файл
    df_submit.to_csv('../../submissions/submission_data/predict_sub_4_model_1.csv', index=False)
