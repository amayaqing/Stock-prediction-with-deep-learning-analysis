import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import nltk

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC, LinearSVC
from sklearn import tree
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

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
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from nltk.probability import FreqDist

import snscrape.modules.twitter as sntwitter
import csv
import pandas as pd
import time

import sys
import re
import string
import json
import os

import datetime
import preprocessor as p
from langdetect import detect

from company import *


def textclean(t):
    '''
    This function cleans the tweets.
    '''
    t = t.lower()  # convert to lowercase
    t = t.replace('\\', "+")
    t = re.sub('\\\\u[0-9A-Fa-f]{4}', '', t)  # remove NON- ASCII characters
    t = re.sub("[0-9]", "", t)  # remove numbers # re.sub("\d+", "", t)
    t = re.sub('[#!"$%&\'()*+,-./:@;<=>?[\\]^_`{|}~]', '', t)  # remove punctuations

    return t

def tweet_process(company_name):
    data = pd.read_csv('data/text/texts{}.csv'.format(company_name), lineterminator='\n', index_col=0)
    rowsize = data.shape[0]

    for i in range(0, rowsize, 1):
        for j in range(1, 61, 1):
            sentence = data.loc[i][j]
            if isinstance(sentence, str):
                print("Raw:", sentence)
                sentence = textclean(sentence)
                print("Cleaned:", sentence)
                if sentence:
                    sentence = remove_emoji(sentence)
                    #                     print("Emoji:",sentence)
                    sentence = textclean(sentence)
                    print("Emoji:", sentence)
                    if len(sentence) > 1:
                        print(len(sentence))
                        lang = langdetect(sentence)
                        if lang == 'en':
                            data.loc[i][j] = sentence
                        else:
                            data.loc[i][j] = np.nan
                    else:
                        data.loc[i][j] = np.nan
                else:
                    data.loc[i][j] = np.nan

    data.to_csv('data/text/cleaned{}.csv'.format(company_name), line_terminator='\n')


def sentiment_analysis(company):
    data = pd.read_csv('data_new/text/cleaned{}.csv'.format(company), lineterminator='\n')
    data = data[data.columns[1:]]
    data['tweets'] = data[data.columns[2:]].apply(lambda x: '. '.join(x.dropna().astype(str)), axis=1)

    pos_list, neg_list = [], []
    sia = SentimentIntensityAnalyzer()
    for i in range(len(data["tweets"])):
        pos, neg = 0, 0
        for text in data["tweets"][i].split('.'):
            res = sia.polarity_scores(text)
            if res['compound'] >= 0.05:
                pos += 1
            elif res['compound'] <= - 0.05:
                neg += 1

        posneg = pos + neg if pos + neg != 0 else 1
        pos_list.append(pos / posneg)
        neg_list.append(neg / posneg)

    df = pd.DataFrame(data={'date': data.date, 'pos': pos_list, 'neg': neg_list})
    df.to_csv('data/rates/rates{}.csv'.format(company), index=False)



def generate_wordcloud(company):
    keywords = company_list + ['tsla', 'stock', 'tickers']

    data = pd.read_csv('data/text/cleaned{}.csv'.format(company), lineterminator='\n')
    data = data[data.columns[1:]]
    data['tweets'] = data[data.columns[2:]].apply(lambda x: '. '.join(x.dropna().astype(str)), axis=1)

    words = data["tweets"][data["tweets"].shape[0] - 1]
    for c in keywords:
        words = words.replace(c.lower(), "")

    # make wordcloud
    wc = WordCloud(
        max_words=200,
        background_color='white',
        #     mask = mask,
        width=2000,
        height=1200
    )

    word_cloud = wc.generate(words)
    word_cloud.background_color = 'white'
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.savefig('assets/wordcloud-{}.png'.format(company), dpi=300)
    plt.show()


def remove_emoji(text):
    p.set_options(p.OPT.URL, p.OPT.EMOJI)
    return p.clean(text)


def langdetect(sentence):
    r=detect(sentence)
    return r





