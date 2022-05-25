# Stock Performance Prediction and Recommendation with Deep Learning Analysis
Course Project for EECS E6895 (Advanced Big Data Analytics and AI) at Columbia University

Author: Jiaqing Chen (jc5657@columbia.edu), Wannuo Sun (ws2591@columbia.edu)

## Project Overview

Stock market prediction is always an attractive and challenging task among investors as it contributes to developing effective strategies in stock trading. This project developed a comprehensive stock prediction system employing Long Short-Term Memory (LSTM), Temporal Convolutional Networks (TCN), and several traditional machine learning models to forecast long-term stock trends and short-term up/down phenomena. Our model takes the publicly available historical stock price as long as topic-related tweets to drive predictions by both history and public knowledge. Empirical experiments show that our models achieve around 80% validation accuracy. We built a dashboard application to offer our prediction results and serve for public usage.

## Setup

To setup and use our dashboard application, you could clone the repository and run app.py ($ python app.py). The dashboard will run on http://127.0.0.1:8050/ on your local computer.

The analysis and prediction will be automaticlaly up-to-date upon running the application, but it will take a while to update. Please be patient when waiting for opening.


## You may need:

```shell script
pip install dash
pip install yfinance
pip install finta

```