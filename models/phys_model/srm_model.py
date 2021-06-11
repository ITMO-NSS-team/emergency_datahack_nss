import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib

import scipy.optimize as optimize
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.max_columns', 15)

font = {'family': 'normal',
        'weight': 'normal',
        'size': 10}

matplotlib.rc('font', **font)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 14),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          # 'axes.weight': 'bold',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
pylab.rcParams.update(params)


def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    '''
    Function taken from
    https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
    '''

    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int) * -1

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new


def model_equation(params, variables):
    '''
    params : np.array
        parameters of the runoff model, params[0] - snow runoff coeff, params[1] - rain_runoff_coeff,
        params[2] - discharge recession coeff, params[3] - degree day factor,
        params[4] - downstream flow coeff;

    variables : tuple
        physical variables of the runoff model
        variables[0] - degree days, variables[1] - fractional snow cover, variables[2] - watershed area,
        variables[3] - rainfall, variables[4] - discharge, variables[5] - upstream point data
    '''
    k = 10000. / 86400.
    return ((params[0] * params[3] * variables[0] * variables[1] +
             params[1] * variables[3]) * variables[2] * k * (1 - params[2]) +
            params[2] * variables[4] +
            params[4] * (variables[4] - variables[5]))


def opt_equation(params, *variables):
    '''
    params : np.array
        parameters of the runoff model, params[0] - snow runoff coeff, params[1] - rain_runoff_coeff,
        params[2] - discharge recession coeff, params[3] - degree day factor,
        params[4] - downstream flow coeff;

    variables : tuple
        physical variables of the runoff model
        variables[0] - degree days, variables[1] - fractional snow cover, variables[2] - watershed area,
        variables[3] - rainfall, variables[4] - discharge, variables[5] - upstream point data

    '''
    k = 10000. / 86400.

    f_py = lambda t_idx: ((params[0] * params[3] * variables[0][t_idx - 1] * variables[1][t_idx - 1] +
                           params[1] * variables[3][t_idx - 1]) * variables[2] * k * (1 - params[2]) +
                          params[2] * variables[4][t_idx - 1] +
                          params[4] * (variables[4][t_idx - 1] - variables[5][t_idx - 1]) - variables[4][t_idx])
    f_vect = np.vectorize(f_py)
    res = np.sum(np.abs(f_vect(np.arange(1, variables[4].size))))
    return res


class DischargeModel(object):
    def __init__(self):
        self.params_list = ['snow_runoff', 'rain_runoff', 'discharge_recession',
                            'degree_day_factor', 'downstream flow']
        self.var_list = ['degree_days', 'frac_snow_cover', 'area', 'rainfall', 'point_discharge', 'upsteam_discharge']

    def get_clusters(self, data, eps=0.5, base_clusters=10):
        self.data_shape = data.shape
        self.clustering_method = 'DBSCAN'
        self.pca = PCA(n_components=2)
        self.pca.fit(data)
        # print(self.pca.explained_variance_ratio_)
        data_transformed = self.pca.transform(data)

        self.scaler = StandardScaler()
        data_transformed_scaled = self.scaler.fit_transform(data_transformed)

        self.clustering = DBSCAN(eps=0.3, min_samples=10).fit(data_transformed_scaled)
        if len(set(self.clustering.labels_)) < 5:
            self.clustering_method = 'KMeans'
            self.clustering = KMeans(n_clusters=base_clusters).fit(data_transformed_scaled)


    def get_params(self, variables, data):
        '''
        variables : tuple
            physical variables of the runoff model
            variables[0] - degree days, variables[1] - fractional snow cover, variables[2] - watershed area,
            variables[3] - rainfall, variables[4] - discharge, variables[5] - upstream point data

        '''
        assert type(variables) == tuple
        self.get_clusters(data)

        bounds = ((0., 1.), (0., 1.),
                  (0., 1.), (0., 0.8),
                  (-10, 10))

        self.cluster_params = {}
        for cluster_label in set(self.clustering.labels_):
            if cluster_label != -1:
                indexes = list(np.where(self.clustering.labels_ == cluster_label))
                var_temp = []
                for var in variables:
                    if isinstance(var, (int, float)):
                        var_temp.append(var)
                    else:
                        var_temp.append(var[indexes])
                self.cluster_params[cluster_label] = optimize.differential_evolution(opt_equation, bounds,
                                                                                     args=var_temp)

    def predict_1day(self, variables, meteodata):
        mdata_transformed = np.dot(meteodata.reshape((1, -1)), self.pca.components_.T)
        mdata_transformed = self.scaler.transform(mdata_transformed)
        if self.clustering_method == 'KMeans':
            variable_cluster = self.clustering.predict(mdata_transformed)
        elif self.clustering_method == 'DBSCAN':
            variable_cluster = dbscan_predict(self.clustering, mdata_transformed)
        return model_equation(self.cluster_params[variable_cluster[0]].x, variables)

    def predict_period(self, variables, meteodata, period=None):
        '''
        period = meteodata.shape[0]
        '''

        if period is None:
            preds = np.empty(meteodata.shape[0])
        else:
            preds = np.empty(period)
            meteodata = np.stack([meteodata for i in np.arange(period)])

        for idx in np.arange(preds.size):
            preds[idx] = self.predict_1day(variables, meteodata[idx, :])
            variables = list(variables)
            variables[4] = preds[idx]
            variables = tuple(variables)

        return preds


def rainfall_convert(total_precip, temps):
    rainfall = np.empty_like(total_precip)
    for idx in np.arange(total_precip.size):
        rainfall[idx] = total_precip[idx] if temps[idx] > 0 else 0

    return rainfall


def get_const_for_3045():
    params = {'area': 897000 - 770000,
              'section_to': 5700,
              'lapse': 0.65,
              'h_mean': 360.0,
              'h_st': 98.0}
    return params


def fit_3045_phys_model(meteo_spring):
    """
    Функция обучает физическую модель для гидрологического поста номер 3045
    """
    # Получим постоянные для данного пункта
    params = get_const_for_3045()
    area = params['area']
    section_to = params['section_to']
    lapse = params['lapse']
    h_mean = params['h_mean']
    h_st = params['h_st']

    dm = DischargeModel()

    # Датафрейм с значениями расходов по станциями
    river = pd.read_csv('../../data/4rd_checkpoint/no_gaps_train.csv', parse_dates=['date'])
    columns = ['date', 'station_id', 'discharge']
    river_3042 = river[river.station_id == 3042][columns]
    river_3036 = river[river.station_id == 3036][columns]
    # Оставляем только значения расходов
    rivers = pd.merge(left=river_3042, right=river_3036, how='inner',
                      on='date', suffixes=('_3042', '_3036'))[['date', 'discharge_3042', 'discharge_3036']]
    # Объединяем датафреймы с расходами и метеопараметрами
    data_full = pd.merge(left=meteo_spring, right=rivers, how='left', on='date')
    data_meteo = data_full[['snow_height', 'snow_coverage_station',
                            'air_temperature', 'relative_humidity',
                            'pressure', 'wind_direction', 'wind_speed_aver',
                            'precipitation']].to_numpy()

    temps = data_full.air_temperature.to_numpy()
    degree_days = temps + lapse * (h_mean - h_st) * 0.01
    temps = data_full.air_temperature.to_numpy()
    degree_days = temps + lapse * (h_mean - h_st) * 0.01
    snow_cover = data_full.snow_coverage_station.to_numpy()
    total_precip = data_full.precipitation.to_numpy()

    # Модифицируем значения осадков
    rainfall = rainfall_convert(total_precip, temps)

    upstream_discharge = data_full.discharge_3036.to_numpy()
    station_discharge = data_full.discharge_3042.to_numpy()
    variables = (degree_days[:section_to], snow_cover[:section_to], area, rainfall[:section_to],
                 station_discharge[:section_to], upstream_discharge[:section_to])
    dm.get_params(variables, data_meteo[:section_to])

    return dm
