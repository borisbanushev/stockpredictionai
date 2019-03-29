# coding=utf-8
import os
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from time_series_modelling.config import data_path
from time_series_modelling.common_misc import parser


def arima_fit(data_series, print_info=True):
    model = ARIMA(data_series, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    if print_info:
        print(model_fit.summary())
    return model_fit


def plot_autocorrelation(series):
    pd.tools.plotting.autocorrelation_plot(series)
    plt.figure(figsize=(10, 7), dpi=80)
    plt.show()


def arima_fit_forecast(data_series):
    """
    It is extending the length of the training data???
    It is a day ahead prediction!
    :return:
    """
    size = int(len(data_series) * 0.66)
    train, test = data_series[0:size], data_series[size:len(data_series)]
    history = [x for x in train]
    predicts = []

    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predicts.append(yhat)
        obs = test[t]
        history.append(obs)

    error = mean_squared_error(test, predicts)
    print(error)
    return test, predicts


def plot_arima_predicts(actual_list, predict_list):
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(actual_list, label="Real")
    plt.plot(predict_list, color='red', label="Predicted")
    plt.xlabel("Days")
    plt.ylabel("USD")
    plt.title("Figure: ARIMA model on GS stock")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    data_file_path = os.path.join(data_path, "Goldman_Sachs_Stock_Price.csv")
    dataset_ex_df = pd.read_csv(data_file_path, header=0, parse_dates=[0], date_parser=parser)

    data = dataset_ex_df["Close"].values
    test_list, predict_list = arima_fit_forecast(data)

    plot_arima_predicts(test_list, predict_list)