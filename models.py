import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
from datetime import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.decomposition import PCA
from pyramid.arima import auto_arima
from dateutil.relativedelta import relativedelta
from dateutil import rrule
from utils import *

def pca_stl_sarima(df, data_test, dept):
    
    name = 'pca_stl_sarima'
    df_inverse = pca_decomposition(df, dept)

    for store in df_inverse.columns:
        try:
            print('store: {}'.format(store))
            ts = df_inverse.loc[:, store]
            seasonal_obj = seasonal_decompose(ts, freq=52)
            seasonality = seasonal_obj.seasonal

            # seasonal adjust
            seasonal_adjusted = ts - seasonality

            # autoarima
            mod = auto_arima(seasonal_adjusted, alpha=0.02, start_p=0, start_q=0, maxiter=100, Start_Q=0, Start_P=0, max_p=3, max_q=3, trend=None, trace=False, m=52, n_fits=3**8, n_jobs=-1)
            fcst_len = get_fcst_len(data=df, store=store, dept=dept, data_test=data_test)
            fcst = mod.predict(fcst_len)

            # reframe
            df_inverse = pivot_df(df, dept)
            start = df_inverse.index[-1] + relativedelta(weeks=1)
            idx = pd.DatetimeIndex(start=start, periods=fcst_len, freq='W-FRI')
            re_idx = [date - relativedelta(weeks=52) for date in idx]
            fcst = pd.Series(fcst, index=re_idx)

            # re-seasonalize
            fcst = pd.DataFrame(seasonality).join(pd.DataFrame(fcst)).dropna().sum(axis=1)
            assert len(fcst) == fcst_len, 'not equal'

            # write test
            fcst.index = idx
            fcst_df = pd.DataFrame(fcst)
            fcst_df['Store'] = store
            fcst_df['Dept'] = dept
            
            send_message(store, name, store=True, fail=False)
            
            yield fcst_df.reset_index()
            
        except:
            print('    fail store {} '.format(store))
            # slack
            send_message(store, name, store=True, fail=True)
            pass
        
        
def run_pca_stl_sarima(df, test):
    name = 'pca_stl_sarima'

    ans = pd.DataFrame()
    dept_ls = list(set([tag.split('_')[1] for tag in test['check'].unique()]))
    dept_ls = list(map(int, dept_ls))

    for dept in dept_ls:
        
        #slack
        text = 'predict dept {} start'.format(dept)
        send_text(name, text)
        
        i=0
        for fcst_yield_df in pca_stl_sarima(df, test, dept):
            i += 1
            ans = ans.append(fcst_yield_df, ignore_index=True)
            ans.to_csv('pca_stl_sarima_new_ans.csv', index=False)
            print('    saved: ', i)
            
            # slack filled
            filled = len(ans)
            text = 'current filled: {}'.format(filled)
            send_text(name, text)

        #slack
        send_message(dept, name, store=False, fail=False)

    return ans

def svd_ets(data, data_test, dept, seasonal='add'):
    name = 'pca_ets'
    
    pca_data = pca_decomposition(data, dept)
    idx = pca_data.columns
    condition = data_test['Dept'] == dept

    for store in data_test[condition]['Store'].unique():
        try:
            print('predict store:', store)
            fcst_len = get_fcst_len(store, dept, data, data_test)
            ts = pca_data.loc[:, store]
            fit = ExponentialSmoothing(ts, seasonal=seasonal, seasonal_periods=52, trend=None).fit(optimized=True, remove_bias=True)
            fcst = fit.forecast(fcst_len)
            fcst_df = pd.DataFrame(fcst)
            fcst_df['Store'] = store
            fcst_df['Dept'] = dept
            
            send_message(store, name, store=True, fail=False)
            yield fcst_df.reset_index()
        
        except:
            print('    fail store {} '.format(store))
            # slack
            send_message(store, name, store=True, fail=True)
            pass

def run_svd_ets(df, test):
    name = 'pca_ets'
    
    ans = pd.DataFrame()
    dept_ls = list(set([tag.split('_')[1] for tag in test['check'].unique()]))
    dept_ls = list(map(int, dept_ls))

    for dept in dept_ls:
        i = 0
        print('predict dept: {}'.format(dept))
        
        # slack
        text = 'predict dept {} start'.format(dept)
        send_text(name, text)
        
        for fcst_yield_df in svd_ets(df, test, dept):
            i += 1
            ans = ans.append(fcst_yield_df, ignore_index=True)
            ans.to_csv('answercsv/complete_svd_ets.csv', index=False)
            print('    saved: ', i)
            
            filled = len(ans)
            text = 'current filled: {}'.format(filled)
            send_text(name, text)

        #slack
        send_message(dept, name, store=False, fail=False)
    return ans

def pca_custom_ets(df, data_test, dept, seasonal='add'):
    
    name = 'pca_custom_ets'
    df_inverse = pca_decomposition(df, dept)

    for store in df_inverse.columns:
        try:
            print('store: {}'.format(store))
            ts = df_inverse.loc[:, store]
            
            # seasonal decompose
            df_seasonality, seasonal_adjustment = custom_seasonal_adjust(ts)

            # ets
            fit = ExponentialSmoothing(seasonal_adjustment, seasonal=seasonal, seasonal_periods=52, trend=None).fit(optimized=True, remove_bias=True)
            
            # forecast
            fcst_len = get_fcst_len(data=df, store=store, dept=dept, data_test=data_test)
            fcst = fit.forecast(fcst_len)
            fcst_df = pd.DataFrame(fcst)
            fcst_df['Store'] = store
            fcst_df['Dept'] = dept
            fcst_df = fcst_df.rename(columns={0:'Weekly_Sales'})
            
            # reseasonalized
            temp = fcst_df.reset_index().assign(week_num=lambda x: x['index'].dt.week).merge(df_seasonality, on='week_num')
            fcst_df.loc[:,'Weekly_Sales'] = temp.apply(lambda x: x['Weekly_Sales_x'] + x['Weekly_Sales_y'], axis=1).values
            send_message(store, name, store=True, fail=False)
            
            yield fcst_df.reset_index().rename(columns={'index':'Date'})
            
        except:
            print('    fail store {} '.format(store))
            # slack
            send_message(store, name, store=True, fail=True)
            pass


def run_pca_custom_ets(df, test):
    name = 'pca_custom_ets'
    
    ans = pd.DataFrame()
    dept_ls = list(set([tag.split('_')[1] for tag in test['check'].unique()]))
    dept_ls = list(map(int, dept_ls))

    for dept in dept_ls:
        i = 0
        print('predict dept: {}'.format(dept))
        
        # slack
        text = 'predict dept {} start'.format(dept)
        send_text(name, text)
        
        for fcst_yield_df in pca_custom_ets(df, test, dept):
            i += 1
            ans = ans.append(fcst_yield_df, ignore_index=True)
            ans.to_csv('pca_custom_ans.csv', index=False)
            print('    saved: ', i)
            
            filled = len(ans)
            text = 'current filled: {}'.format(filled)
            send_text(name, text)

        #slack
        send_message(dept, name, store=False, fail=False)
    return ans


def pca_custom_sarima(df, data_test, dept):
    
    name = 'pca_custom_sarima'
    df_inverse = pca_decomposition(df, dept)

    for store in df_inverse.columns:
        try:
            print('store: {}'.format(store))
            ts = df_inverse.loc[:, store]
            
            # seasonal decompose
            df_seasonality, seasonal_adjustment = custom_seasonal_adjust(ts)
            mod = auto_arima(seasonal_adjustment, m=52, alpha=0.01, n_jobs=-1, trace=True, trend=None)

             # forecast
            fcst_len = get_fcst_len(data=df, store=store, dept=dept, data_test=data_test)
            fcst = mod.predict(fcst_len)
            fcst_df = pd.DataFrame(fcst)

            # re-frame
            start = seasonal_adjustment.index[-1] + relativedelta(weeks=1)
            idx = pd.DatetimeIndex(start=start, freq='W-FRI', periods=fcst_len)

            fcst_df.index = idx
            fcst_df['Store'] = store
            fcst_df['Dept'] = dept
            fcst_df = fcst_df.rename(columns={0:'Weekly_Sales'})

            # re-seasonalized
            temp = fcst_df.reset_index().assign(week_num=lambda x: x['index'].dt.week).merge(df_seasonality, on='week_num')
            fcst_df.loc[:,'Weekly_Sales'] = temp.apply(lambda x: x['Weekly_Sales_x'] + x['Weekly_Sales_y'], axis=1).values
            
            yield fcst_df.reset_index().rename(columns={'index':'Date'})
            
            # slack
            send_message(store, name, store=True, fail=False)

            
        except:
            print('    fail store {} '.format(store))
            # slack
            send_message(store, name, store=True, fail=True)
            pass


def run_pca_custom_sarima(df, test):
    name = 'pca_custom_sarima'
    
    ans = pd.DataFrame()
    dept_ls = list(set([tag.split('_')[1] for tag in test['check'].unique()]))
    dept_ls = list(map(int, dept_ls))

    for dept in dept_ls:
        i = 0
        print('predict dept: {}'.format(dept))
        
        # slack
        text = 'predict dept {} start'.format(dept)
        send_text(name, text)
        
        for fcst_yield_df in pca_custom_sarima(df, test, dept):
            i += 1
            ans = ans.append(fcst_yield_df, ignore_index=True)
            ans.to_csv('pca_custom_sarima_ans.csv', index=False)
            print('    saved: ', i)
            
            filled = len(ans)
            text = 'current filled: {}'.format(filled)
            send_text(name, text)

        #slack
        send_message(dept, name, store=False, fail=False)
    return ans        