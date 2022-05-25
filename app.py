# -*- coding: utf-8 -*-
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly
import pathlib
import pandas as pd


from pages import (
    overview,
    company_detail
)

from utils import *
from stock_utils import *
from trend_prediction import *
from update_all import *

# get relative data folder
PATH = pathlib.Path(__file__).parent


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    suppress_callback_exceptions=True
)
app.title = "AI Stock Analytics Dashboard"
server = app.server

# Describe the layout/ UI of the app
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

# Update page
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/dash-stock-prediction/company_detail":
        return company_detail.create_layout(app)
    elif pathname == "/dash-financial-report/full-view":
        return (
            overview.create_layout(app),
            company_detail.create_layout(app),
        )
    else:
        return overview.create_layout(app)


# Company_detail, performance
@app.callback(
    Output("performance_graph", "figure"),
    [
        Input("toggle-rangeslider", "value"),
        Input("company-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def display_candlestick(value, company, start_date, end_date):
    data = get_stock_data(company, start_date, end_date)
    fig = go.Figure(go.Candlestick(
        x = data['Date'],
        open = data['Open'],
        high = data['High'],
        low = data['Low'],
        close = data['Close']
    ))

    fig.update_layout(
        # template="plotly_white",
        xaxis_rangeslider_visible='slider' in value
    )

    return fig

# Company_detail, trend prediction
@app.callback(
    Output("trend_prediction_graph", "figure"),
    [
        Input("company-filter", "value"),
    ],
)
def display_trend_prediction(company):

    res = pd.read_csv("data/trend-prediction-{}.csv".format(company))
    data = pd.read_csv("data/stock-{}.csv".format(company))
    train_data = data[(len(data) // 20) * 19:]

    fig = plotly.tools.make_subplots(specs=[[{"secondary_y":False}]])
    fig.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Open'], name="History"), secondary_y=False, )
    fig.add_trace(go.Scatter(x=res['Date'], y=res['Pred'], name="Prediction", mode="lines"), secondary_y=False, )
    fig.update_layout(
        autosize=False, width=900, height=500,
        title_text=company,
        # template="plotly_white"
    )
    fig.update_xaxes(title_text="year")
    fig.update_yaxes(title_text="prices", secondary_y=False)

    return fig

# Company_detail, updown_table
@app.callback(
    [Output("updown_table", "children")],
    [Input("company-filter", "value")],
)
def update_updown_table(company):
    # train_data, res = trend_prediction_lstm(company, start_date, end_date)
    df = pd.read_csv("data/updown-prediction-{}.csv".format(company))

    return [html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +
        # Body
        [html.Tr([
            html.Td(df.iloc[i][col]) for col in df.columns
        ]) for i in range(df.values.shape[0])]
    )]

# wordcloud
@app.callback(
    [Output("wordcloud", "children")],
    [Input("company-filter", "value")],
)
def update_wordcloud(company):
    # train_data, res = trend_prediction_lstm(company, start_date, end_date)

    src = app.get_asset_url("wordcloud-{}.png".format(company))
    return [
        html.Img(
            src=src,
            className="cloud",
        )
    ]



start_time = "2018-01-01"
end_time = date.today().strftime("%Y-%m-%d")

print("Generating today's analysis......")
update(start_time, end_time)
print("Update completed. You can open the dashboard now.")


if __name__ == "__main__":
    app.run_server(debug=False)



