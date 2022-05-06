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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout, Embedding
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical

import datetime
from datetime import date
from stock_utils import *
from trend_prediction import *
from risk_prediction import *
from company import *
from twitter_process import *


def updown_to_int(updown):
    return 1 if updown == 'Up' else 0

def update(start_time, end_time):

    # update stock price
    for company in company_list:
        data = get_stock_data(company, start_time, end_time)
        data.to_csv('data/stock-{}.csv'.format(company), index=False)

    for company in company_list:
        get_label(company, start_time, end_time)

    # update trend
    for company in company_list:
        predict_trend_lstm(company, start_time, end_time)


    # update up/down
    # for company in company_list:
    #     sentiment_analysis(company)
    # for company in company_list:
    #     risk_prediction_all(company, start_time, end_time)
    # for company in company_list:
    #     generate_wordcloud(company)

    # update overview
    predicted_trend_list = []
    updown_list = []
    current_price_list = []
    for company in company_list:
        data = yf.download(company, start=start_time, end=end_time, interval="1d")
        current_price = data['Open'].values[-1]

        df = pd.read_csv('data/trend-prediction-{}.csv'.format(company))
        diff = (df['Pred'].values[-1] - df['Pred'].values[0]) / df['Pred'].values[0]
        predicted_trend = round(diff, 6) * 100

        df = pd.read_csv('data/updown-prediction-{}.csv'.format(company))
        pred = [updown_to_int(x) for x in df['Up/Down prediction'].values]
        acc = df['Accuracy'].values
        valid_pred = [pred[x] for x in range(len(pred)) if acc[x] >= 0.8]
        final_pred = 'Up' if np.mean(valid_pred) >= 0.5 else 'Down'

        current_price_list.append(current_price)
        predicted_trend_list.append(predicted_trend)
        updown_list.append(final_pred)

        df_overview = {
            'Stock': company_list,
            'Current Price': current_price_list,
            'Predicted Trend in 15 days': predicted_trend_list,
            'Up/Down in next day': updown_list
        }
        df_overview = pd.DataFrame(data=df_overview)
        df_overview = df_overview.sort_values(by=['Predicted Trend in 15 days'], ascending=False)
        df_overview = df_overview.reset_index(drop=True)
        df_overview.to_csv('data/overview.csv'.format(company),
                           header=list(df_overview.columns.values), index=False)




