from dash import dcc
from dash import html
import plotly.graph_objs as go
import plotly.express as px

from utils import Header, make_dash_table

import pandas as pd
import pathlib
import datetime
import yfinance as yf

import sys
sys.path.append("..")
from stock_utils import *
from company import *

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

df_overview= pd.read_csv(DATA_PATH.joinpath("overview.csv"))

stock = {}
for company in company_list:
    stock[company] = pd.read_csv(DATA_PATH.joinpath("stock-{}.csv".format(company)))



# up down pie
updown_list = list(df_overview['Up/Down in next day'].values)
up = updown_list.count('Up')
df_updown_pie = pd.DataFrame(
    data={'updown': [up, len(updown_list)-up]},
    index=['Up', 'Down']
)

# trend pie
predicted_trend_list = list(df_overview['Predicted Trend in 15 days'].values)
count_2 = 0
count_1 = 1
count_0 = 2
for x in predicted_trend_list:
    if x > 1:
        count_2 += 1
    elif x < -1:
        count_0 += 1
    else:
        count_1 += 1
df_trend_pie = pd.DataFrame(
    data={'updown': [count_2, count_1, count_0]},
    index=['Great Up', 'Flat', 'Bad Down']
)

def create_layout(app):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [
                    # Overview
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Dashboard Summary"),
                                    html.H6("(The analysis is based on date {})".format(datetime.date.today())),
                                    html.Br([]),
                                    html.P(
                                        "\
                                    Here, we provide you with a comprehensive stock analysis and predictions \
                                    with focus on famous international companies. We applied the state-of-art \
                                    machine learning techniques to predict the general future trend and  \
                                    next-day up/down situation. The analysis is based on both fundemantal analysis \
                                    and some event-driven factors by the means of twitter collection. \
                                    The overview page lists the conclusive analysis and visualizes the general market \
                                    situation. The company stock details list the detailed situation for each company,\
                                    including which models we used, which twitter content is related to, etc. \
                                    Currently, we provide analysis for S&P 500 and twelve famous technical companies\
                                    : Google (GOOG), Meta (FB), Amazon (AMZN), Microsoft (MSFT), Apple (AAPL), IBM (IBM), DELL (DELL), SONY (SONY), Intel (INTC), Tencent (TCEHY), HP (HPQ), Cisco (CSCO)",
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),
                                ],
                                className="product",
                            )
                        ],
                        className="row",
                    ),
                    # stock recommendation
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Br([]),
                                    html.H6("Stock Analysis and Prediction Overview", className="subtitle padded"),
                                    html.Div(
                                        [html.Table(
                                            # Header
                                            [html.Tr([html.Th(col) for col in df_overview.columns])] +
                                            # Body
                                            [html.Tr([
                                                html.Td(df_overview.iloc[i][col]) for col in df_overview.columns
                                            ]) for i in range(df_overview.values.shape[0])]
                                        )],
                                        style={"overflow-x": "auto"},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),


                    # Index Plot
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("The Whole Picture", className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-4",
                                        figure={
                                            "data": [
                                                go.Scatter(
                                                    x=stock["^GSPC"]["Date"],
                                                    y=stock["^GSPC"]["Open"],
                                                    line={"color": "#97151c"},
                                                    mode="lines",
                                                    name="S&P 500",
                                                ),
                                                go.Scatter(
                                                    x=stock["GOOG"]["Date"],
                                                    y=stock["GOOG"]["Open"],
                                                    line={"color": "#b5b5b5"},
                                                    mode="lines",
                                                    name="GOOG",
                                                ),
                                                go.Scatter(
                                                    x=stock["FB"]["Date"],
                                                    y=stock["FB"]["Open"],
                                                    line={"color": "#862983"},
                                                    mode="lines",
                                                    name="FB",
                                                ),
                                                go.Scatter(
                                                    x=stock["AMZN"]["Date"],
                                                    y=stock["AMZN"]["Open"],
                                                    line={"color": "#ff8da3"},
                                                    mode="lines",
                                                    name="AMAZ",
                                                ),
                                                go.Scatter(
                                                    x=stock["MSFT"]["Date"],
                                                    y=stock["MSFT"]["Open"],
                                                    line={"color": "#ffd7cf"},
                                                    mode="lines",
                                                    name="AAPL",
                                                ),
                                                go.Scatter(
                                                    x=stock["AAPL"]["Date"],
                                                    y=stock["AAPL"]["Open"],
                                                    line={"color": "#ffc66e"},
                                                    mode="lines",
                                                    name="MSFT",
                                                ),
                                                go.Scatter(
                                                    x=stock["IBM"]["Date"],
                                                    y=stock["IBM"]["Open"],
                                                    line={"color": "#d46c4e"},
                                                    mode="lines",
                                                    name="IBM",
                                                ),
                                                go.Scatter(
                                                    x=stock["DELL"]["Date"],
                                                    y=stock["DELL"]["Open"],
                                                    # line={"color": "#d46c4e"},
                                                    mode="lines",
                                                    name="DELL",
                                                ),
                                                go.Scatter(
                                                    x=stock["SONY"]["Date"],
                                                    y=stock["SONY"]["Open"],
                                                    # line={"color": "#d46c4e"},
                                                    mode="lines",
                                                    name="SONY",
                                                ),
                                                go.Scatter(
                                                    x=stock["INTC"]["Date"],
                                                    y=stock["INTC"]["Open"],
                                                    # line={"color": "#d46c4e"},
                                                    mode="lines",
                                                    name="INTC",
                                                ),
                                                go.Scatter(
                                                    x=stock["TCEHY"]["Date"],
                                                    y=stock["TCEHY"]["Open"],
                                                    # line={"color": "#d46c4e"},
                                                    mode="lines",
                                                    name="TCEHY",
                                                ),
                                                go.Scatter(
                                                    x=stock["HPQ"]["Date"],
                                                    y=stock["HPQ"]["Open"],
                                                    # line={"color": "#d46c4e"},
                                                    mode="lines",
                                                    name="HPQ",
                                                ),
                                                go.Scatter(
                                                    x=stock["CSCO"]["Date"],
                                                    y=stock["CSCO"]["Open"],
                                                    # line={"color": "#d46c4e"},
                                                    mode="lines",
                                                    name="CSCO",
                                                ),
                                            ],
                                            "layout": go.Layout(
                                                autosize=True,
                                                width=1000,
                                                height=400,
                                                font={"family": "Raleway", "size": 14},
                                                margin={
                                                    "r": 30,
                                                    "t": 30,
                                                    "b": 30,
                                                    "l": 30,
                                                },
                                                showlegend=True,
                                                titlefont={
                                                    "family": "Raleway",
                                                    "size": 10,
                                                },
                                                xaxis={
                                                    "autorange": True,
                                                    # "range": [
                                                    #     "2007-12-31",
                                                    #     "2018-03-06",
                                                    # ],
                                                    "rangeselector": {
                                                        "buttons": [
                                                            {
                                                                "count": 1,
                                                                "label": "1Y",
                                                                "step": "year",
                                                                "stepmode": "backward",
                                                            },
                                                            {
                                                                "count": 3,
                                                                "label": "3Y",
                                                                "step": "year",
                                                                "stepmode": "backward",
                                                            },
                                                            {
                                                                "label": "All",
                                                                "step": "all",
                                                            },
                                                        ]
                                                    },
                                                    "showline": True,
                                                    "type": "date",
                                                    "zeroline": False,
                                                },
                                                yaxis={
                                                    "autorange": True,
                                                    # "range": [
                                                    #     18.6880162434,
                                                    #     278.431996757,
                                                    # ],
                                                    "showline": True,
                                                    "type": "linear",
                                                    "zeroline": False,
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                # className="twelve columns",
                                className="wrapper",
                            )
                        ],
                        className="row ",
                    ),
                    # market situation
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Distribution of Up/Down in next-day"], className="subtitle padded"
                                    ),

                                    # px.pie(df_updown_pie, values='updown', names=df_updown_pie.index),
                                    dcc.Graph(
                                        id='pie-updown',
                                        figure={
                                            "data": [
                                                {
                                                    "labels": df_updown_pie.index,
                                                    "values": df_updown_pie['updown'],
                                                    "type": "pie",
                                                    "marker": {"line": {"color": "white", "width": 1}},
                                                    "hoverinfo": "label",
                                                    "textinfo": "label",
                                                }
                                            ],
                                        }
                                    ),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        ["Distribution of Trend Up/Down in future 15 days"],
                                        className="subtitle padded",
                                    ),
                                    # html.Table(make_dash_table(df_hist_prices)),
                                    dcc.Graph(
                                        id='pie-trend',
                                        figure={
                                            "data": [
                                                {
                                                    "labels": df_trend_pie.index,
                                                    "values": df_trend_pie['updown'],
                                                    "type": "pie",
                                                    "marker": {"line": {"color": "white", "width": 1}},
                                                    "hoverinfo": "label",
                                                    "textinfo": "label",
                                                }
                                            ],
                                        }
                                    ),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
