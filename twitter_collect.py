import snscrape.modules.twitter as sntwitter
import csv
import pandas as pd
import time
import datetime

from twitter_process import *
from company import *
from stock_utils import *


# --------------------------- crawl tweets ------------------------------

now = time.time()

data = pd.DataFrame(columns=['date'])
for i in range(0, 60, 1):
    data.insert(data.shape[1], i, 0)


for company_name in company_list:
    idx = -1
    for year in range(2018, 2023, 1):
        for month in range(1, 13, 1):
            if month in [1, 3, 5, 7, 8, 10, 12]:
                for day in range(1, 32, 1):
                    idx += 1
                    start_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
                    end_date = getNextDay(year, month, day)

                    if company_name == "^GSPC":
                        keyword = "stock"
                    else:
                        keyword = company_name
                    search_content = "{} + since:{} until:{} -filter:links -filter:replies".format(keyword,
                                                                                                   start_date, end_date)

                    for j, tweet in enumerate(sntwitter.TwitterSearchScraper(search_content).get_items()):
                        data.loc[idx, 'date'] = start_date
                        data.loc[idx, j] = tweet.content
                        if j == 59:
                            break
            elif month in [4, 6, 9, 11]:
                for day in range(1, 31, 1):
                    idx += 1
                    start_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
                    end_date = getNextDay(year, month, day)
                    search_content = "{} + since:{} until:{} -filter:links -filter:replies".format(company_name,
                                                                                                   start_date, end_date)

                    for j, tweet in enumerate(sntwitter.TwitterSearchScraper(search_content).get_items()):
                        data.loc[idx, 'date'] = start_date
                        data.loc[idx, j] = tweet.content
                        if j == 59:
                            break
            elif month == 2:
                if year == 2020:
                    for day in range(1, 30, 1):
                        idx += 1
                        start_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
                        end_date = getNextDay(year, month, day)
                        search_content = "{} + since:{} until:{} -filter:links -filter:replies".format(company_name,
                                                                                                       start_date,
                                                                                                       end_date)

                        for j, tweet in enumerate(sntwitter.TwitterSearchScraper(search_content).get_items()):
                            data.loc[idx, 'date'] = start_date
                            data.loc[idx, j] = tweet.content
                            if j == 59:
                                break
                else:
                    for day in range(1, 29, 1):
                        idx += 1
                        start_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
                        end_date = getNextDay(year, month, day)
                        search_content = "{} + since:{} until:{} -filter:links -filter:replies".format(company_name,
                                                                                                       start_date,
                                                                                                       end_date)

                        for j, tweet in enumerate(sntwitter.TwitterSearchScraper(search_content).get_items()):
                            data.loc[idx, 'date'] = start_date
                            data.loc[idx, j] = tweet.content
                            if j == 59:
                                break

    data.to_csv('data/text/texts{}.csv'.format(company_name), line_terminator='\n')

print("time: ", time.time() - now)

for company_name in company_list:
    tweet_process(company_name)
    sentiment_analysis(company_name)
    generate_wordcloud(company_name)