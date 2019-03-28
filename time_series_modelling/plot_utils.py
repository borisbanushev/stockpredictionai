# coding=utf-8
import datetime
import matplotlib.pyplot as plt


def plot_raw_price(dataset_ex_df):
    plt.figure(figsize=(14, 5), dpi=100)
    plt.plot(dataset_ex_df["Date"], dataset_ex_df["Close"], label="Google stock")
    plt.vlines(datetime.date(2016, 4, 20), 0, 270, linestyles='--', colors='gray', label="Train/Test data cut-off")
    plt.xlabel("Date")
    plt.ylabel('USD')
    plt.title("Figure 2: Goldman Sachs stock price")
    plt.legend()
    plt.show()


def plot_technical_indicators(dataset, last_days, target_col):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0 - last_days

    dataset = dataset.iloc[-last_days:, :]
    x = range(3, dataset.shape[0])
    x_ = list(dataset.index)

    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'], label="MA 7", color='g', linestyle="--")
    plt.plot(dataset[target_col], label="Closing Price", color='b')
    plt.plot(dataset['ma21'], label="MA 21", color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Goldman Sachs - last {} days.'.format(last_days))
    plt.ylabel("USD")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("MACD")
    plt.plot(dataset['MACD'], label="MACD", linestyle='--')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyle='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyle='--')
    plt.plot(dataset["log_momentum"], label="Momentum", color='b', linestyle='-')
    plt.legend()
    plt.show()