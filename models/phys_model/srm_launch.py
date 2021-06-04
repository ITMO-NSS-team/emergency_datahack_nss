import os
import pandas as pd
import numpy as np
from models.phys_model.srm_model import DischargeModel


if __name__ == '__main__':
    cwd = os.getcwd()
    meteo_snow = pd.read_csv('../../data/meteo_data/no_gap_1day/no_gap_meteo_1day_int_3045.csv', parse_dates=['date'])
    meteo_press = pd.read_csv('../../data/meteo_data/no_gap_3hour/no_gap_meteo_3hour_int_3045_press.csv', parse_dates=['date'])
    meteo_wind = pd.read_csv('../../data/meteo_data/no_gap_3hour/no_gap_meteo_3hour_int_3045_wind.csv', parse_dates=['date'])
    meteo_integrated = pd.merge(meteo_snow, meteo_press, on=['station_id', 'date'])
    meteo_integrated = pd.merge(meteo_integrated, meteo_wind, on=['station_id', 'date'])
    start_month = 4
    end_month = 9
    mask = (meteo_integrated['date'].dt.month >= start_month) & (meteo_integrated['date'].dt.month <= end_month)
    meteo_integrated_spring = meteo_integrated.loc[mask]
    meteo_integrated_spring = meteo_integrated_spring.dropna()
    meteo_integrated_spring.head()
    dm = DischargeModel()

    area = 0.0
    river = pd.read_csv('../../data/4rd_checkpoint/no_gaps_train.csv', parse_dates=['date'])
    columns = ['date', 'station_id', 'discharge']
    river_3042 = river[river.station_id == 3042][columns]
    river_3036 = river[river.station_id == 3036][columns]
    rivers = pd.merge(left=river_3042, right=river_3036, how='inner',
                      on='date', suffixes=('_3042', '_3036'))[['date', 'discharge_3042', 'discharge_3036']]

    data_full = pd.merge(left=meteo_integrated_spring, right=rivers, how='left', on='date')

    data_meteo = data_full[['snow_height', 'snow_coverage_station',
                            'air_temperature', 'relative_humidity',
                            'pressure', 'wind_direction', 'wind_speed_aver',
                            'precipitation']].to_numpy()

    lapse = 0.65
    h_mean = 360.0
    h_st = 98.0
    temps = data_full.air_temperature.to_numpy()
    degree_days = temps + lapse * (h_mean - h_st) * 0.01
    temps = data_full.air_temperature.to_numpy()
    degree_days = temps + lapse * (h_mean - h_st) * 0.01
    snow_cover = data_full.snow_coverage_station.to_numpy()
    total_precip = data_full.precipitation.to_numpy()

    rainfall = np.empty_like(total_precip)
    for idx in np.arange(total_precip.size):
        rainfall[idx] = total_precip[idx] if temps[idx] > 0 else 0

    area = 897000 - 770000

    section_to = 5700
    upstream_discharge = data_full.discharge_3036.to_numpy()
    station_discharge = data_full.discharge_3042.to_numpy()
    variables = (degree_days[:section_to], snow_cover[:section_to], area, rainfall[:section_to],
                 station_discharge[:section_to], upstream_discharge[:section_to])
    dm.get_params(variables, data_meteo[:section_to])
