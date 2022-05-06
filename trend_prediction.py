import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.dates as dates

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from finta import TA
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import train_test_split

import datetime


def LSTM_trend_model(dim):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(dim, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model


def predict_trend_lstm(company, start_time, end_time):
    data = yf.download(company, start=start_time, end=end_time, interval="1d")
    data['Date'] = data.index

    n = len(data)
    train_data = data[(n // 20) * 10:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data['Open'].values.reshape(-1, 1))

    prediction_days = 30
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data) - 5):  ######
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])  ###### predict 5 days after

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = LSTM_trend_model(x_train.shape[1])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)

    x_test = []
    pred_period = 15
    for day in range(prediction_days, prediction_days - pred_period, -1):
        x_test.append(scaled_data[-day:, 0])

    np.append(x_test[-1], [1])

    y_pred = [data['Open'].values[-1]]
    for i in range(pred_period):
        x_test[i] = np.array(x_test[i])
        x_test[i] = np.reshape(x_test[i], (1, 30, 1))
        predicted_prices = model.predict(x_test[i])
        for j in range(i + 1, pred_period):
            x_test[j] = np.append(x_test[j], [predicted_prices])
        #         print(x_test[j].shape)

        y_pred.append(scaler.inverse_transform(predicted_prices)[0][0])

    now = datetime.datetime(int(end_time[:4]), int(end_time[5:7]), int(end_time[8:10]))
    date = [str(now - datetime.timedelta(days=1))[:10]]
    for i in range(pred_period):
        date.append(str(now + datetime.timedelta(days=i))[:10])

    res = {'Date': date, 'Pred': y_pred}
    res = pd.DataFrame(data=res)
    res = res.set_index(['Date'])
    res['Date'] = res.index
    res.to_csv('data/trend-prediction-{}.csv'.format(company), index=False)


# def trend_prediction_lstm(company, start_time, end_time):
#     data = yf.download(company, start=start_time, end=end_time, interval="1d")
#     data['Date'] = data.index
#
#     n = len(data)
#     train_data = data[(n // 20) * 10:]
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(train_data['Open'].values.reshape(-1, 1))
#
#     prediction_days = 30
#     x_train = []
#     y_train = []
#
#     for x in range(prediction_days, len(scaled_data) - 5):  ######
#         x_train.append(scaled_data[x - prediction_days:x, 0])
#         y_train.append(scaled_data[x, 0])  ###### predict 5 days after
#
#     x_train, y_train = np.array(x_train), np.array(y_train)
#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#
#     model = LSTM_model(x_train.shape[1])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#     model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
#
#     x_test = []
#     pred_period = 15
#     for day in range(prediction_days, prediction_days - pred_period, -1):
#         x_test.append(scaled_data[-day:, 0])
#
#     np.append(x_test[-1], [1])
#
#     y_pred = [data['Open'].values[-1]]
#     for i in range(pred_period):
#         x_test[i] = np.array(x_test[i])
#         x_test[i] = np.reshape(x_test[i], (1, 30, 1))
#         predicted_prices = model.predict(x_test[i])
#         for j in range(i + 1, pred_period):
#             x_test[j] = np.append(x_test[j], [predicted_prices])
#         #         print(x_test[j].shape)
#
#         y_pred.append(scaler.inverse_transform(predicted_prices)[0][0])
#
#     now = datetime.datetime(int(end_time[:4]), int(end_time[5:7]), int(end_time[8:10]))
#     date = [str(now - datetime.timedelta(days=1))[:10]]
#     for i in range(pred_period):
#         date.append(str(now + datetime.timedelta(days=i))[:10])
#
#     res = {'Date': date, 'Pred': y_pred}
#     res = pd.DataFrame(data=res)
#     res = res.set_index(['Date'])
#     res['Date'] = res.index
#
#     res.to_csv('dash-stock-prediction/data/trend-prediction-{}.csv'.format(company), index=False)
#
#     train_data = data[(n // 20) * 19:]
#
#     # verify
#     #     verify_data = yf.download(company, start=date[0], end=date[-1], interval="1d")
#     #     verify_data['Date'] = verify_data.index
#
#     return train_data, res
