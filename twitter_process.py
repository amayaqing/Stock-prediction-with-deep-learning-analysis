import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

import re
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


def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    if sentiment_dict['compound'] >= 0.05:
        return 1

    elif sentiment_dict['compound'] <= - 0.05:
        return -1

    else:
        return 0


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





