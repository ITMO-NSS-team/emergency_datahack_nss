import pandas as pd
import numpy as np

from datetime import date
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 18, 7


df = pd.read_csv('../first_data/track_2_package/meteo_1month.csv')
df['date'] = pd.to_datetime(df['date'])

print(f' Названия переменных: {df.columns}')
print(df.head(5))

for station in df['station_id'].unique():
    station_df = df[df['station_id'] == station]

    plt.plot(station_df['date'], station_df['precipitation_observed'], label='precipitation_observed')
    plt.legend(fontsize=15)
    plt.show()
