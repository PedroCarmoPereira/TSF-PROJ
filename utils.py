import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from itertools import product
from statsmodels.tsa.stattools import acf, pacf

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

    selected_data['lagged_prec'] = selected_data['prec'].shift(1)
    selected_data['lagged_tmed'] = selected_data['tmed'].shift(1)
    selected_data['lagged_tmax'] = selected_data['tmed'].shift(1)
    selected_data['lagged_tmin'] = selected_data['tmed'].shift(1)
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

import numpy as np
from itertools import product
from statsmodels.tsa.stattools import acf, pacf


def sarima_candidates_from_acf(
    y,
    m=None,          # seasonal period, e.g. 12 for monthly data. If None -> no seasonal part.
    max_lag=None,    # max lag for ACF/PACF
    max_p=3,
    max_q=3,
    max_P=1,
    max_Q=1,
    d=0,
    D=0,
    alpha=0.05,
    max_short_lag=5,  # how far to look for non-seasonal p,q
):
    """
    Suggest SARIMA(p,d,q)(P,D,Q,m) candidates based on ACF/PACF.

    Parameters
    ----------
    y : array-like
        1D time series (already differenced if needed).
    m : int or None
        Seasonal period (e.g. 12 for monthly data). If None, no seasonal part is suggested.
    max_lag : int or None
        Maximum lag for ACF/PACF. If None, uses min(40, len(y)//2).
    max_p, max_q : int
        Maximum non-seasonal AR/MA order to consider (upper bounds, but we also cap by max_short_lag and m-1).
    max_P, max_Q : int
        Maximum seasonal AR/MA order to consider. In this implementation we only use 0 or 1 (textbook Box–Jenkins).
    d, D : int
        Already-used non-seasonal and seasonal differencing orders.
    alpha : float
        Significance level for ACF/PACF cut-off (default 0.05, ~95% bounds).
    max_short_lag : int
        Maximum lag to look at for non-seasonal dynamics. Typically 3–5 is enough.

    Returns
    -------
    result : dict
        {
          "acf": np.ndarray,
          "pacf": np.ndarray,
          "crit": float,
          "p_candidates": list[int],
          "q_candidates": list[int],
          "P_candidates": list[int],
          "Q_candidates": list[int],
          "candidates": list[tuple]   # (p, d, q, P, D, Q, m)
        }
    """
    y = np.asarray(y)
    n = len(y)

    if max_lag is None:
        max_lag = min(40, n // 2)

    # --- 1. Compute ACF and PACF ---
    acf_vals = acf(y, nlags=max_lag, fft=True)
    pacf_vals = pacf(y, nlags=max_lag, method="ywm")

    # approximate 95% significance bounds: ±1.96 / sqrt(n)
    crit = 1.96 / np.sqrt(n)

    # --- 2. Non-seasonal candidates (p, q) ---

    # limit how far we look for non-seasonal structure
    # and never let q reach the seasonal lag m
    max_p_lag = min(max_p, max_short_lag)
    if m is not None and m > 1:
        max_q_lag = min(max_q, max_short_lag, m - 1)
    else:
        max_q_lag = min(max_q, max_short_lag)

    # p ~ AR terms from PACF at small lags
    p_candidates = [0]
    for k in range(1, max_p_lag + 1):
        if abs(pacf_vals[k]) > crit:
            p_candidates.append(k)
    p_candidates = sorted(set(p_candidates))
    if len(p_candidates) == 1:  # only [0]
        p_candidates.append(1)  # at least allow AR(1)
    # optionally cap how many we keep
    p_candidates = p_candidates[:3]

    # q ~ MA terms from ACF at small lags
    q_candidates = [0]
    for k in range(1, max_q_lag + 1):
        if abs(acf_vals[k]) > crit:
            q_candidates.append(k)
    q_candidates = sorted(set(q_candidates))
    if len(q_candidates) == 1:  # only [0]
        q_candidates.append(1)  # at least allow MA(1)
    q_candidates = q_candidates[:3]

    # --- 3. Seasonal candidates (P, Q) ---

    P_candidates = [0]
    Q_candidates = [0]

    if m is not None and m > 1 and m <= max_lag:
        # how many seasonal lags we can even look at (m, 2m, 3m, ...)
        max_seasonal_k = max_lag // m

        # P from PACF at seasonal lags: m, 2m, 3m, ...
        for j in range(1, min(max_P, max_seasonal_k) + 1):
            lag = j * m
            if abs(pacf_vals[lag]) > crit:
                P_candidates.append(j)

        # Q from ACF at seasonal lags: m, 2m, 3m, ...
        for j in range(1, min(max_Q, max_seasonal_k) + 1):
            lag = j * m
            if abs(acf_vals[lag]) > crit:
                Q_candidates.append(j)

    P_candidates = sorted(set(P_candidates))
    Q_candidates = sorted(set(Q_candidates))
    # --- 4. Build candidate list ---

    candidates = []
    s = m or 0
    for p, q, P, Q in product(p_candidates, q_candidates, P_candidates, Q_candidates):
        # Avoid invalid MA overlap: if Q>0, don't let q >= s (non-seasonal MA up to seasonal lag)
        if s > 0 and Q > 0 and q >= s:
            continue
        candidates.append((p, d, q, P, D, Q, s))

    return {
        "acf": acf_vals,
        "pacf": pacf_vals,
        "crit": crit,
        "p_candidates": p_candidates,
        "q_candidates": q_candidates,
        "P_candidates": P_candidates,
        "Q_candidates": Q_candidates,
        "candidates": candidates,
        "max_lag": max_lag
    }
