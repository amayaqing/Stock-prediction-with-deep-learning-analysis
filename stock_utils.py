import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from finta import TA
import numpy as np
import pandas as pd
from datetime import date

from company import *

start_time = "2018-12-31"
end_time = date.today().strftime("%Y-%m-%d")

def get_stock_data(company, start_time, end_time):
    data = yf.download(company, start=start_time, end=end_time, interval="1d")
    data['Date'] = data.index
    return data


def get_label(company, start_time, end_time):
    data = yf.download(company, start=start_time, end=end_time, interval="1d")
    data['Date'] = data.index

    hist_data = []
    label = []
    date = []
    for i in range(20, data.shape[0] - 1):
        hist_data.append([data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i], data['Adj Close'][i],
                          data['Volume'][i]])
        if data['Open'][i + 1] >= data['Open'][i]:
            label.append(1)
        else:
            label.append(0)
        date.append(data['Date'][i])

    df_label = pd.DataFrame(data={'Date': date, 'Label': label})
    df_label.to_csv('data/label/label-{}.csv'.format(company), index=False)