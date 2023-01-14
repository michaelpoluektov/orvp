import glob
import pandas as pd
import numpy as np
import os
import gc
import json

with open('SETTINGS.json', 'r') as f:
  data = json.load(f)
data_dir = data["RAW_DATA_DIR"]
clean_data_dir = data["CLEAN_DATA_DIR"]
all_train_book = glob.glob(f'{data_dir}book_train.parquet/**/*.parquet')
all_test_book = glob.glob(f'{data_dir}book_test.parquet/**/*.parquet')

from math import sqrt
from joblib import Parallel, delayed
from sklearn import preprocessing, model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import NearestNeighbors
from itertools import chain
from itertools import permutations

from numpy.random import seed
seed(42)
from scipy.linalg import lstsq

def get_time_ids(is_train = True):
    files = all_train_book if is_train else all_test_book
    return np.sort(pd.concat([pd.read_parquet(file, columns = ['time_id']) for file in files], ignore_index = True)['time_id'].unique())

time_splits = [0, 100, 200, 300, 400, 500, 600]

def log_return(series):
    return np.log(series).diff()

# Function to calculate first WAP
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

# Function to count unique elements of a series
def count_unique(series):
    return len(np.unique(series))

# RQQ replaced rolling window with convolve for performance
def realized_quadpower_quarticity(series):
    convolved = np.exp(np.convolve(np.log(series.to_numpy()), [1, 1, 1, 1], mode='valid'))
    return (np.sum(convolved) * (convolved.shape[0]+3) * (np.pi**2))/4

def tendency(price, vol):    
    df_diff = np.diff(price.to_numpy())
    val = (df_diff/price.to_numpy()[1:])*100
    power = np.sum(val*vol.to_numpy()[1:])
    return(power)

def iqr(series): 
    return np.percentile(series.to_numpy(),75) - np.percentile(series.to_numpy(),25)

def abs_diff(series): 
    return np.median(np.abs(series.to_numpy() - np.mean(series.to_numpy())))

def energy(series): 
    return np.mean(series.to_numpy()**2)

def f_max(series): 
    return np.sum(series.to_numpy() > np.mean(series.to_numpy()))

def f_min(series): 
    return np.sum(series.to_numpy() < np.mean(series.to_numpy()))

def df_max(series): 
    return np.sum(np.diff(series.to_numpy()) > 0)

def df_min(series): 
    return np.sum(np.diff(series.to_numpy()) < 0)

def list_names(dic, prefix = '', splits = []):
    return list(chain(*[[f'{prefix}{key}_{func.__name__}{t}' for func in lis] for key, lis in dic.items() for t in ['']+splits]))

def fix_offsets(book_df, trade_df):
    offsets = book_df.groupby(['time_id']).agg({'seconds_in_bucket':'min'})
    offsets.columns = ['offset']
    book_df = book_df.join(offsets, on='time_id')
    trade_df = trade_df.join(offsets, on='time_id')
    book_df.seconds_in_bucket = book_df.seconds_in_bucket - book_df.offset
    trade_df.seconds_in_bucket = trade_df.seconds_in_bucket - trade_df.offset
    book_df.drop(columns=['offset'], inplace=True)
    trade_df.drop(columns=['offset'], inplace=True)
    return book_df, trade_df

def fix_offsets_book(data_df):
    offsets = data_df.groupby(['time_id']).agg({'seconds_in_bucket':'min'})
    offsets.columns = ['offset']
    data_df = data_df.join(offsets, on='time_id')
    data_df.seconds_in_bucket = data_df.seconds_in_bucket - data_df.offset
    return data_df

def ffill(data_df):
    data_df=data_df.set_index(['time_id', 'seconds_in_bucket'])
    data_df = data_df.reindex(pd.MultiIndex.from_product([data_df.index.levels[0], np.arange(0,600)], names = ['time_id', 'seconds_in_bucket']), method='ffill')
    return data_df.reset_index()

def get_book_files(is_train = True):
    df_files = pd.DataFrame(
        {'book_path': all_train_book if is_train else all_test_book}) \
        .eval('stock_id = book_path.str.extract("stock_id=(\d+)").astype("int")', engine='python')
    df_prices = pd.concat(Parallel(n_jobs=4, verbose=51)(delayed(calc_prices)(r) for _, r in df_files.iterrows()))
    return df_prices.pivot(index = 'time_id', columns = 'stock_id', values = 'price')

def get_dir_stats(arr):
    coefs, err, _, _ = lstsq(np.array([np.ones(len(arr)), np.arange(len(arr))]).transpose(), arr)
    return np.array([coefs[0] + (len(arr)/2)*coefs[1], coefs[1], err])

# Function to preprocess book data (for each stock id)
def book_preprocessor(df, stock_id, time_ids):
    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    # Calculate log returns
    df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
    df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)
    # Calculate spread
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    
    # Dict for aggregations
    create_feature_dict = {
        'wap1': [realized_quadpower_quarticity, count_unique],
        'wap2': [count_unique],
        'log_return1': [np.sum, np.min, np.max],
        'log_return2': [np.sum],
        'price_spread':[np.mean, np.max],
        'price_spread2':[np.mean, np.max],
        'total_volume':[np.sum],
    }
    
    create_feature_dict_time = {
        'log_return1': [realized_volatility],
        'log_return2': [realized_volatility],
    }
    
    df_feature = df.groupby(['time_id']).agg(create_feature_dict).reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]
    df_feature = df_feature.set_index('time_id_').reindex(pd.Series(time_ids))
    df_features = []
    for start, stop in zip(time_splits[:-1], time_splits[1:]):
        df_tmpp = df[(df['seconds_in_bucket'] >= start) & (df['seconds_in_bucket'] < stop)].groupby(['time_id']).agg(create_feature_dict_time).reset_index()
        df_tmpp.columns = ['_'.join(col) for col in df_tmpp.columns]
        df_tmpp = df_tmpp.set_index('time_id_').reindex(pd.Series(time_ids))
        df_tmpp = df_tmpp.fillna(0)
        df_features.append(df_tmpp[list_names(create_feature_dict_time)].to_numpy().transpose())
    df_features = np.stack(df_features, axis = 2)
    df_features = np.apply_along_axis(get_dir_stats, 2, df_features)
    for name, vals in zip(list_names(create_feature_dict_time), df_features):
        for attr, col in zip(['intercept', 'slope', 'error'], vals.transpose()):
            df_feature[f'{name}_{attr}'] = col
    
    # Create row_id so we can merge
    df_feature.reset_index(inplace = True)
    df_feature['row_id'] = df_feature['index'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['index'], axis = 1, inplace = True)
    return df_feature

# Function to preprocess trade data (for each stock id)
def trade_preprocessor(df, stock_id, time_ids):
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    # Dict for aggregations
    create_feature_dict = {
        'price':[energy, abs_diff],
        'log_return':[realized_volatility],
    }
    
    create_feature_dict_time = {
        'size':[np.sum],
        'order_count':[np.sum]
    }
    
    df_feature = df.groupby(['time_id']).agg(create_feature_dict).reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]
    df_feature = df_feature.set_index('time_id_').reindex(pd.Series(time_ids))
    df_features = []
    for start, stop in zip(time_splits[:-1], time_splits[1:]):
        df_tmpp = df[(df['seconds_in_bucket'] >= start) & (df['seconds_in_bucket'] < stop)].groupby(['time_id']).agg(create_feature_dict_time).reset_index()
        df_tmpp.columns = ['_'.join(col) for col in df_tmpp.columns]
        df_tmpp = df_tmpp.set_index('time_id_').reindex(pd.Series(time_ids))
        df_tmpp = df_tmpp.fillna(0)
        df_features.append(df_tmpp[list_names(create_feature_dict_time)].to_numpy().transpose())
    df_features = np.stack(df_features, axis = 2)
    df_features = np.apply_along_axis(get_dir_stats, 2, df_features)
    for name, vals in zip(list_names(create_feature_dict_time), df_features):
        for attr, col in zip(['intercept', 'slope', 'error'], vals.transpose()):
            df_feature[f'{name}_{attr}'] = col
    df_feature.reset_index(inplace = True)

    # Get the stats for different windows
    df_feature = df_feature.add_prefix('trade_')
    df_feature['row_id'] = df_feature['trade_index'].apply(lambda x:f'{stock_id}-{x}')
    df_feature.drop(['trade_index'], axis = 1, inplace = True)
    return df_feature

def get_time_stats(df):
    vol_cols = [f'log_return1_realized_volatility_{t}' for t in ['intercept', 'slope', 'error']]
    vol_cols += ['trade_order_count_sum_intercept', 'trade_order_count_sum_slope']
    df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix('_' + 'time')
    df = df.merge(df_time_id, how = 'left', left_on = ['time_id'], right_on = ['time_id__time'])
    df.drop(['time_id__time'], axis = 1, inplace = True)
    return df

# Funtion to make preprocessing function in parallel (for each stock id)
def preprocessor(list_stock_ids, is_train = True):
    list_time_ids = get_time_ids(is_train)
    def for_joblib(stock_id):
        # Train
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        # Test
        else:
            file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)
            
        book_df, trade_df = fix_offsets(pd.read_parquet(file_path_book), pd.read_parquet(file_path_trade))
        # Preprocess book and trade data and merge them
        df_tmp = pd.merge(book_preprocessor(book_df, stock_id, list_time_ids), trade_preprocessor(trade_df, stock_id, list_time_ids), on = 'row_id', how = 'left')
        # Return the merge dataframe
        return df_tmp
    
    # Use parallel api to call paralle for loop
    df = Parallel(n_jobs = -1, verbose = 51)(delayed(for_joblib)(stock_id) for stock_id in list_stock_ids)
    # Concatenate all the dataframes that return from Parallel
    df = pd.concat(df, ignore_index = True)
    return df

def generate_df(is_train):
    df = pd.read_csv(f'{data_dir}{"train" if is_train else "test"}.csv')
    df['row_id'] = df['stock_id'].astype(str) + '-' + df['time_id'].astype(str)
    print(f'Our {"training" if is_train else "testing"} set has {df.shape[0]} rows')
    stock_ids = df['stock_id'].unique()
    # Preprocess them using Parallel and our single stock id functions
    pre_ = preprocessor(stock_ids, is_train = is_train)
    df = df.merge(pre_, on = ['row_id'], how = 'left')
    df = get_time_stats(df)
    df.drop(columns = ['trade_size_sum_error', 'trade_size_sum_slope'], inplace = True)
    return df

if __name__ == "__main__":
    generate_df(True).to_parquet(f'{clean_data_dir}orvp-features.parquet')
