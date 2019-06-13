import warnings
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.decomposition import PCA
from pmdarima.arima import auto_arima
from dateutil.relativedelta import relativedelta
from datetime import *
from dateutil import rrule
from slacker import Slacker
warnings.filterwarnings('ignore')

def init_df(df, test):
    """
    return: df, test
    """
    print('loading data'+ '...')
    #df_train = df.groupby(['Store', 'Date']).agg(np.mean).drop(columns=['Dept', 'IsHoliday'])
    #df_t = df_train.unstack().T
    df['Date'] = pd.to_datetime(df['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    #test.drop(columns='IsHoliday', inplace=True)
    test['Weekly_Sales'] = 0

    df['check'] = df.apply(lambda x: str(x['Store']) + '_' + str(x['Dept']), axis=1)
    test['check'] = test.apply(lambda x: str(x['Store']) + '_' + str(x['Dept']), axis=1)

    print('loading data'+ '...' * 2)

    def supplement_data(df, test, store, dept):
        """
        - Add the ID which is in test but not in train
        - Fill df with weekly_sales of consecutive date to meet consistency
        """

        df_s = pd.DataFrame({'Store': store,
                'Dept': dept,
                'Date': all_date,
                'Weekly_Sales': 0,
                'check': str(store) + '_' + str(dept)})
        return df_s

    
    # make complete data
    all_date = df['Date'].unique()
    all_test_check = test['check'].unique()
    all_train_check = df['check'].unique()
    check_tag = np.where(~np.isin(all_test_check, all_train_check))[0]
    need_to_add = all_test_check[check_tag]

    print('loading data'+ '...' * 3)

    for tag in need_to_add:
        store = int(tag.split('_')[0])
        dept = int(tag.split('_')[1])
        df = df.append(supplement_data(df, test, store, dept), ignore_index=True).fillna(0)
    
    return df, test


def get_value(data, store=1, dept=1):
    """
    return values of specific store and dept
    """
    c = data["Store"] == store
    c2 = data["Dept"] == dept
    return data[c&c2].reset_index(drop=True)

def pivot_df(df, dept=1):
    """
    pivot dataframe and fillna(0)
    """
    c = df['Dept'] == dept
    df_pivot = df[c].pivot(index='Date', columns='Store', values='Weekly_Sales').fillna(0)
    start = df_pivot.index[0]
    end = df_pivot.index[-1]
    idx = pd.DatetimeIndex(start=start, end=end, freq='W-FRI')
    df_pivot = df_pivot.merge(pd.DataFrame(idx).rename(columns={0:'Date'}), how='outer', on='Date').fillna(0)
    df_pivot = df_pivot.sort_index()
    return df_pivot.set_index('Date')

def reframe_df(previous_df, processed_data):
    """
    convert array to pivot_table
    """
    idx = previous_df.index
    col = previous_df.columns
    df = pd.DataFrame(data=processed_data, index=idx, columns=col)
    return df

def pca_decomposition(data, dept, n_components=12):
    """
    PCA deomposition according to the Dept
    """
    try:
        df_svd = pivot_df(data, dept)
        pca = PCA(n_components=n_components)
        df_low = pca.fit_transform(df_svd)
        df_inverse = pca.inverse_transform(df_low)

        # re-frame
        df_inverse = reframe_df(previous_df=df_svd, processed_data=df_inverse)
        return df_inverse

    except:
        # if pca fail,
        return pivot_df(data, dept)
    
def get_fcst_len(store, dept, data, data_test):
    """
    Get the length of periods to forecast
    """
    def weeks_between(start_date, end_date):
        weeks = rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date)
        return weeks.count()
    

    c = data_test['Store'] == store
    c2 = data_test['Dept'] == dept
    start = pivot_df(data, dept).index[-1] + relativedelta(weeks=1)
    end = data_test[c&c2]['Date'].iloc[-1]
    fcst_len = weeks_between(start, end)
    return fcst_len

def custom_seasonal_adjust(denoise):
    """
    - Average sales of each week num
    - It is used to do diff to adjust seasonality
    """
    df_adjust = pd.DataFrame()
    df_adjust[0] = denoise.values
    df_adjust[1] = denoise.shift(-52).values
    df_adjust[2] = denoise.shift(-104).values
    seasonality = df_adjust.mean(axis=1)[:52]

    start_idx = denoise.index[0]
    df_seasonality = seasonality.append(seasonality, ignore_index=True).append(seasonality[:39], ignore_index=True)
    idx = pd.DatetimeIndex(start=start_idx, freq='W-FRI', periods=len(df_seasonality))
    df_seasonality.index = idx
    
    seasonal_adjust = (denoise - df_seasonality).dropna()
    df_seasonality = df_seasonality.reset_index().\
    assign(week_num = lambda x: x['index'].dt.week).\
    drop_duplicates('week_num').\
    drop(columns='index').rename(columns={0:'Weekly_Sales'})
    
    return df_seasonality, seasonal_adjust

def fill_test_form(data, data_test, fcst, store, dept):
    """
    fill test form with fcst data
    """
    c = data_test['Store'] == store
    c2 = data_test['Dept'] == dept
    fcst = pd.DataFrame(fcst).rename(columns={0: 'Weekly_Sales'})
    try:
        fcst_for_test = data_test[c&c2].set_index('Date').join(fcst, on='Date', how='left', lsuffix='_0', rsuffix='_1').drop(columns='Weekly_Sales_0').Weekly_Sales_1
    except:
        start = pivot_df(data, dept).index[-1] + relativedelta(weeks=1)
        idx = pd.DatetimeIndex(start=start, periods=len(fcst), freq='W-FRI')
        fcst.index = idx                       
        fcst_for_test = data_test[c&c2].set_index('Date').join(fcst, on='Date', how='left', lsuffix='_0', rsuffix='_1').drop(columns='Weekly_Sales_0').Weekly_Sales_1
    data_test.loc[c&c2,'Weekly_Sales'] = fcst_for_test.values
    c = np.where(data_test['Weekly_Sales'] > 0)[0]
    return data_test 


def send_message(arg, name, store=False, fail=False):
    """
    send slack message of the result of modeling
    """
    with open('token.pkl', 'rb') as f:
        token = pickle.load(f)
    slack = Slacker(token)
    attachments_dict = dict()
    attachments_dict['pretext'] = name
    
    if store:
        if fail:
            text = '! fail: store {}'.format(arg)
        else: 
            text = 'store {} success'.format(arg)
    else: 
        if fail:
            text = '! fail: dept {}'.format(arg)
        else:
            text = 'dept {} success'.format(arg)
        
    
    attachments_dict['text'] = text
    slack.chat.post_message('#random', text=None, attachments=[attachments_dict])

    
def send_text(name, text):
    with open('token.pkl', 'rb') as f:
        token = pickle.load(f)
    slack = Slacker(token)
    attachments_dict = dict()
    attachments_dict['pretext'] = name
    attachments_dict['text'] = text
    slack.chat.post_message('#random', text=None, attachments=[attachments_dict])

# submission related code below:
def make_Id_check(df):
    """
    make 'check' columns: 'store_dept' 
    - eg. store1 dept1 = '1_1'
    """
    df['check'] = df.apply(lambda x: str(x['Store']) + '_' + str(x['Dept']), axis=1)
    df = df.drop(columns=['Store', 'Dept'])
    return df

def fill_test(answer_filename):
    name = answer_filename
    
    # get answer csv
    ans = pd.read_csv(name)
    try:
        ans['Date'] = pd.to_datetime(ans['Date'])
    except:
        ans = ans.rename(columns={'index':'Date'})
        ans['Date'] = pd.to_datetime(ans['Date'])

    ans = make_Id_check(ans)
    
    # get test form
    test = pd.read_csv('test.csv')
    test['Date'] = pd.to_datetime(test['Date'])
    test = test.drop(columns=['IsHoliday'])
    test = make_Id_check(test)
    
    # fill test form
    test = test.merge(ans, how='left', on=['check', 'Date']).fillna(0)
    
    return test


def shift_2_5(sub):
        
    def modify_sub(sub):
        sub['Date'] = sub['Id'].apply(lambda x: x.split('_')[2])
        sub['check'] = sub['Id'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1])
        sub['Date'] = pd.to_datetime(sub['Date'])
        sub['Week_num'] = sub['Date'].dt.week
        return sub

    # prepare apply shift to submission file
    modified_df = modify_sub(sub)
    
    c52 = modified_df['Week_num'] == 52
    c51 = modified_df['Week_num'] == 51
    
    len_ = len(modified_df['check'].unique())
    len_ = int(len_ * 0.1)
    
    print('total number of IDs: ', len_); i = 0;
    
    for Id in modified_df['check'].unique():
        i += 1
        if not i % len_:
            print('complete: ', int(i / (len_ * 10) * 100), '%')
            
        c = modified_df['check'] == Id
        try:
            val1 = modified_df.loc[c&c51].Weekly_Sales.values[0] * (2.5/7)
            val2 = modified_df.loc[c&c52].Weekly_Sales.values[0] * (2.5/7)
            
            modified_df.loc[c&c51, 'Weekly_Sales'] = modified_df.loc[c&c51, 'Weekly_Sales'] - val1 + val2
            modified_df.loc[c&c52, 'Weekly_Sales'] = modified_df.loc[c&c52, 'Weekly_Sales'] - val2 + val1
        
        except:
            pass

    return modified_df.drop(columns=['Date', 'check', 'Week_num'])  