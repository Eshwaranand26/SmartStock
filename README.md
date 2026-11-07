# SmartStock

Predicting the next day closing price of a stock using SimpleRNN and LSTM.

I made this after learning RNNs and LSTMs. Stock prices are a time series (every day depends on the days before it), so I wanted to see how well these models can actually predict the next day price.

## What's in the notebook

- Download L&T (LT.NS) daily prices from 2002 till today using yfinance
- Basic understanding of the data (shape, info, missing values)
- EDA: closing price, volume, distribution, 100 and 200 day moving averages, correlation heatmap
- Split the data by time: first 80% of days for training, last 20% for testing (no shuffling, otherwise future data goes into training)
- Use the last 100 days to predict the next day
- Train a SimpleRNN and an LSTM and compare them
- A small predictive system at the end where you give any stock name and it predicts the next day close

## Results

To check if the model is actually learning something, I compared it with a simple baseline: "tomorrow's price = today's price".

| Model | RMSE (in rupees) |
|---|---|
| Baseline (yesterday price) | 48.52 |
| SimpleRNN | 50.87 |
| LSTM | 51.93 |

Both models come very close to the baseline but don't beat it. So the model is mostly following the previous day's price. This is pretty normal for stock prices, because the next day price mostly depends on today's price and on news, which is not in the data.

## Things I learned

- First I used MinMaxScaler on the price, but the prices in the last few years are way higher than anything in the training data. The test values went above 1 and the LSTM predicted much lower than the actual price (RMSE was around 202).
- So instead I divided every 100 day window by its own last day price. That way the values stay near 1 for any year or any stock, and the error dropped to around 50.
- Always compare with a baseline. Without it the prediction chart looks perfect, but it's basically copying yesterday's price.

## How to run

Open `SmartStock.ipynb` in Google Colab or Jupyter and run all the cells. The data downloads automatically, no CSV needed.

If you're running it locally:

```
pip install numpy pandas matplotlib seaborn yfinance tensorflow
```

`stock_lstm_model.keras` is the trained LSTM model. The last section of the notebook loads it and predicts the next day close for any stock (use `.NS` for NSE and `.BO` for BSE).

## To do

- Try GRU and compare with LSTM
- Use more features (Open, High, Low, Volume) and not only Close
- Hyperparameter tuning (units, window size, epochs)
- Predict whether the price goes up or down instead of the exact price
- Predict the next 30 days instead of only the next day

This is just a learning project, not financial advice.
