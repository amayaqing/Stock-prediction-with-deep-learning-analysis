# import dash_core_components as dcc
import dash
from dash import dcc
from dash import html
# import dash_html_components as html
from utils import Header, make_dash_table
from dash.dependencies import Input, Output
import pandas as pd
import pathlib
from dash_table import DataTable

from company import *


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

df_current_prices = pd.read_csv(DATA_PATH.joinpath("df_current_prices.csv"))
df_hist_prices = pd.read_csv(DATA_PATH.joinpath("df_hist_prices.csv"))
df_avg_returns = pd.read_csv(DATA_PATH.joinpath("df_avg_returns.csv"))
df_after_tax = pd.read_csv(DATA_PATH.joinpath("df_after_tax.csv"))
df_recent_returns = pd.read_csv(DATA_PATH.joinpath("df_recent_returns.csv"))
df_graph = pd.read_csv(DATA_PATH.joinpath("df_graph.csv"))
df_updown = pd.read_csv(DATA_PATH.joinpath("updown-prediction-GOOG.csv"))

data = pd.read_csv(DATA_PATH.joinpath("stock-GOOG.csv"))
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
data.sort_values("Date", inplace=True)


def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 2
            html.Div(
                [
                    html.Div(
                        children=[
                            html.H6(
                                ["Company Stock Interested"]
                            ),
                            # html.Div(children="Company Stock Interested", className="subtitle padded"),
                            dcc.Dropdown(
                                id="company-filter",
                                options=[
                                    {"label": company, "value": company}
                                    # for avocado_type in data.type.unique()
                                    for company in company_list
                                ],
                                value="GOOG",
                                clearable=False,
                                searchable=False,
                                className="dropdown",
                            ),
                        ],
                        className="row ",
                    ),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("Performance", className="subtitle padded"),
                                    dcc.DatePickerRange(
                                        id="date-range",
                                        min_date_allowed=data.Date.min().date(),
                                        max_date_allowed=data.Date.max().date(),
                                        start_date=data.Date.min().date(),
                                        end_date=data.Date.max().date(),
                                    ),
                                    dcc.Checklist(
                                        id='toggle-rangeslider',
                                        options=[{'label': 'Include Rangeslider',
                                                  'value': 'slider'}],
                                        value=['slider']
                                    ),
                                    dcc.Graph(id="performance_graph"),
                                ],
                                className="wrapper",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("Prediction Price", className="subtitle padded"),
                                    dcc.Graph(id="trend_prediction_graph"),
                                ],
                                className="wrapper",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "Up/Down Prediction under different models"
                                        ],
                                        className="subtitle padded",
                                    ),

                                    html.Div(
                                        [
                                            html.Tr([html.Th(col) for col in df_updown.columns]),
                                        ],

                                            id="updown_table",
                                            style={"overflow-x": "auto"},

                                    ),
                                ],
                                className=" twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 5
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Related Tweets"],
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        html.Img(
                                            src=app.get_asset_url("wordcloud-GOOG.png"),
                                            className="cloud",
                                        ),
                                        id="wordcloud",
                                    ),

                                ],
                                className=" twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )



