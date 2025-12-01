import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

DATA_DIR = 'data'
PREC_FILE = 'prec-Mainland-raw.csv'
TEMP_FILE = 'temp-Mainland-raw.csv'

DATA_START = "1995-01-01"
DATA_END = "2020-01-01"

def load_prec():
    prec_data = pd.read_csv(os.path.join(DATA_DIR, PREC_FILE))
    prec_data = prec_data.melt(id_vars=["year"], var_name="month_str", value_name="prec")
    month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
                'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    prec_data["month"] = prec_data["month_str"].map(month_map)
    prec_data["date"] = pd.to_datetime(dict(year=prec_data["year"], 
                                        month=prec_data["month"], 
                                        day=1))
    
    return prec_data

def load_temp():
    temp_data = pd.read_csv(os.path.join(DATA_DIR, TEMP_FILE))
    temp_data['month'] = temp_data['date'].str.extract(r'([0-9]{2})')
    temp_data['year'] = temp_data['date'].str.extract(r'([0-9]{4})')
    temp_data['month'] = pd.to_numeric(temp_data['month'])
    temp_data['year'] = pd.to_numeric(temp_data['year'])
    temp_data['date'] = pd.to_datetime(temp_data['date'], format='%m/%Y')

    return temp_data

def load_data():
    prec_data = load_prec()
    temp_data = load_temp()
    # Merge Dataframes
    full_data = pd.merge(temp_data, prec_data[["date", "prec"]], 
                  on="date", how="inner")
    # Feature Engineering
    full_data['tdiff'] = full_data['tmax'] - full_data['tmin']

    selected_data = full_data[full_data['date'].between(DATA_START, DATA_END)]
    selected_data['log_diff'] = np.log(selected_data['tdiff'])
    selected_data = selected_data.set_index('date', drop=False)
    return selected_data

def check_stationarity(timeseries, regression='ct'):
    adf_val = adfuller(timeseries, regression=regression, autolag='AIC')
    p_value = adf_val[1]
    print(f'ADF Statistic: {adf_val[0]}')
    print(f'p-value: {p_value}')
    if p_value < 0.05:
        print('Stationary')
    kpss_val = kpss(timeseries, regression=regression, nlags="auto")
    p_value = kpss_val[1]
    print(f'KPSS Statistic: {kpss_val[0]}')
    print(f'p-value: {p_value}')
    if p_value > 0.05:
        print('Stationary')