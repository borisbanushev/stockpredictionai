# coding=utf-8
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def fourier_transformation(data, target_column):
    """
    As you see in Figure 3 the more components from the Fourier transform we use the closer the approximation
    function is to the real stock price (the 100 components transform is almost identical to the original function
    - the red and the purple lines almost overlap). We use Fourier transforms for the purpose of extracting long-
    and short-term trends so we will use the transforms with 3, 6, and 9 components. You can infer that the
    transform with 3 components serves as the long term trend!
    :param data:
    :param target_column:
    :return:
    """
    close_fft = np.fft.fft(np.asarray(data[target_column].tolist()))
    fft_df = pd.DataFrame({"fft": close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df["angle"] = fft_df['fft'].apply(lambda x: np.angle(x))
    return fft_df


def plot_ft(fft_df, data_df, target_column):
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9, 100]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        plt.plot(np.fft.ifft(fft_list_m10), label="Fourier transform with {} components.".format(num_))
    plt.plot(data_df[target_column], label="Real")
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('Figure 3: Goldman Sachs (close) stock prices & Fourier transforms')
    plt.legend()
    plt.show()