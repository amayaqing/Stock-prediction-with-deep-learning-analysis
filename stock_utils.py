import yfinance as yf
import pandas as pd
from datetime import date
import datetime

import snscrape.modules.twitter as sntwitter

from company import *
from twitter_process import *

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



def getNextDay(y, m, d):
    gDate = datetime.datetime(y, m, d)
    nextday = gDate + datetime.timedelta(days=1)
    nDate = '{:%Y-%m-%d}'.format(nextday)
    return nDate



def date_prediction(company_name, year, month, day):
    start_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
    end_date = getNextDay(year, month, day)

    if company_name == "TCEHY":
        keyword = "Tencent"
    elif company_name == "^GSPC":
        keyword = "stock"
    else:
        keyword = company_name
    search_content = "{} + since:{} until:{} -filter:links -filter:replies".format(keyword, start_date, end_date)

    texts_list = []
    sentence_list = []
    pos = 0
    neg = 0
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_content).get_items()):
        texts_list.append(tweet.content)
        if i == 59:
            break
    # print(len(texts_list))
    for j in range(0, min(60, len(texts_list)), 1):
        sentence = texts_list[j]
        if sentence:
            sentence = textclean(sentence)
            if sentence:
                sentence = remove_emoji(sentence)
                sentence = textclean(sentence)
                sentence_list.append(sentence)

                if sentiment_scores(sentence) == 1:
                    pos += 1
                elif sentiment_scores(sentence) == -1:
                    neg += 1

    if pos == 0 and neg == 0:
        pos_rate = 0
        neg_rate = 0
    else:
        pos_rate = pos / (pos + neg)
        neg_rate = neg / (pos + neg)

    s = ' '.join(sentence_list)
    return s, pos_rate, neg_rate