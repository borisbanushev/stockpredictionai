# coding=utf-8
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import mxnet as mx

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb

from time_series_modelling.config import data_path
from time_series_modelling.plot_utils import plot_raw_price, plot_technical_indicators
from time_series_modelling.fourier_transforms import fourier_transformation, plot_ft
warnings.filterwarnings("ignore")





def get_technical_indicators(dataset, target_col):
    # creat 7 and 21 daya moving average
    dataset['ma7'] = dataset[target_col].rolling(window=7).mean()
    dataset['ma21'] = dataset[target_col].rolling(window=21).mean()

    # create MACD: Provides exponential weighted functions.
    dataset['26ema'] = dataset[target_col].ewm(span=26).mean()
    dataset['12ema'] = dataset[target_col].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])

    # create Bollinger Bands
    dataset['20sd'] = dataset[target_col].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + dataset['20sd'] * 2
    dataset['lower_band'] = dataset['ma21'] - dataset['20sd'] * 2

    # create expoential moving average
    dataset['ema'] = dataset[target_col].ewm(com=0.5).mean()
    # create momentum
    dataset['log_momentum'] = np.log(dataset[target_col] - 1)
    return dataset





if __name__ == "__main__":
    context = mx.cpu()
    model_ctx = mx.cpu()
    mx.random.seed(1719)

    data_file_path = os.path.join(data_path, "Goldman_Sachs_Stock_Price.csv")
    dataset_ex_df = pd.read_csv(data_file_path, header=0, parse_dates=[0], date_parser=parser)
    # dataset_ex_df = dataset_ex_df[dataset_ex_df["Date"] >= datetime(2010, 2, 1)]
    # dataset_ex_df.reset_index(drop=True, inplace=True)
    print(dataset_ex_df.head())
    print('There are {} number of days in the dataset.'.format(dataset_ex_df.shape[0]))

    #plot_raw_price(dataset_ex_df)

    num_training_days = int(dataset_ex_df.shape[0] * .7)
    print('Number of training days: {}. Number of test days: {}.'.format(num_training_days,
                                                                         dataset_ex_df.shape[0] - num_training_days))

    dataset_TI_df = get_technical_indicators(dataset_ex_df, "Close")
    print(dataset_TI_df.head())

    #plot_technical_indicators(dataset_TI_df, 400, target_col="Close")

    data_FT = dataset_ex_df[["Date", "Close"]]
    fft_df = fourier_transformation(data_FT, target_column="Close")
    plot_ft(fft_df, data_FT, target_column="Close")