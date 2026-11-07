# SmartStock

Trying to predict the next day of a stock in three ways: the closing price, up or down, and big move or quiet day.

I made this after learning ML and RNNs/LSTMs. Stock prices are a time series (every day depends on the days before it), so I wanted to see how well these models can actually predict the next day. Everything is in one notebook, `SmartStock.ipynb`, split into three parts.

The data is L&T (LT.NS) daily prices from Yahoo Finance using yfinance, plus the Nifty 50 index for parts 2 and 3. In every part the split is by time: the first 80% of days for training and the last 20% for testing. No shuffling, otherwise future days go into training.

## Part 1: next day closing price

Uses the last 100 days of closing prices to predict the next day close, with a SimpleRNN and an LSTM.

To check if the model is actually learning something, I compared it with a simple baseline: "tomorrow's price = today's price".

| Model | RMSE (in rupees) |
|---|---|
| Baseline (yesterday price) | 48.56 |
| SimpleRNN | 52.24 |
| LSTM | 51.25 |

Both models come close to the baseline but don't beat it, so they are mostly following the previous day's price. The random seed is fixed, so the numbers come out the same every run.

First I used MinMaxScaler on the price, but the prices in the last few years are way higher than anything in the training data. The test values went above 1 and the LSTM predicted much lower than the actual price (RMSE was around 202). So instead I divided every 100 day window by its own last day price. That way the values stay near 1 for any year or any stock.

## Part 2: up or down

Since the exact price didn't beat the baseline, I tried a different question: will tomorrow's close be higher or lower than today's? This makes it a classification problem.

Features made from the data: daily return, 5 day return, Nifty return, volume change, day range (High - Low), and price compared to the 20 and 50 day moving averages.

| Model | Test Accuracy |
|---|---|
| Baseline: always down | 48.12% |
| Baseline: always up | 51.88% |
| Logistic Regression | 49.19% |
| Decision Tree | 50.81% |
| Random Forest | 47.79% |
| XGBoost | 48.22% |
| SimpleRNN | 50.27% |
| LSTM | 49.52% |

All of them are around 50%, basically a coin toss, and none beat just saying "always up". The correlation heatmap already showed this: none of the features have any real correlation with tomorrow's direction.

## Part 3: big move or quiet day

Direction looked random, but the size of the move might not be. Big move days usually come in groups (after a big day, the next day is more likely to be big too). So the question here is: will tomorrow be a big move day or a quiet day?

A big move day means tomorrow's range (High - Low) is more than the stock's average range over the last 20 days. I compare with the stock's own recent average instead of one fixed number, because the last few years were much calmer than 2008 or 2020. With a fixed number almost every test day was "quiet". The features are ratios against the last 20 days too: today's range, the 5 day average range, today's move, short vs long volatility, Nifty's move and volume.

| Model | Test Accuracy |
|---|---|
| Baseline: always quiet | 58.82% |
| Baseline: same as today | 57.97% |
| **Logistic Regression** | **63.21%** |
| Decision Tree | 61.28% |
| Random Forest | 61.60% |
| XGBoost | 62.14% |
| SimpleRNN | 57.75% |
| LSTM | 60.53% |

This is the only part where the models beat the baselines. Logistic Regression is about 4% better, and the biggest coefficients were for the recent range ratios, which matches the idea that big days come together. SimpleRNN was the only model below the baselines.

It isn't perfect. It only catches 29% of the big days, but when it says "big move" it's right about 61% of the time.

## Summary

| Part | Question | Best baseline | Best model | Beats baseline? |
|---|---|---|---|---|
| 1 | next day close price | yesterday price (RMSE 48.56) | LSTM (RMSE 51.25) | No |
| 2 | up or down | always up (51.88%) | Decision Tree (50.81%) | No |
| 3 | big move or quiet | always quiet (58.82%) | Logistic Regression (63.21%) | Yes |

Things I learned:

- The price and direction of the next day can't be predicted from past prices alone. The size of the move can, a little.
- Simple ML models did as well as or better than SimpleRNN and LSTM on this kind of data.
- Always compare with a baseline. Without it the prediction chart in part 1 looks perfect, but the model is basically copying yesterday's price.

## Files

- `SmartStock.ipynb`: all the code (data, EDA, the three parts, saving the models)
- `stock_lstm_model.keras`: LSTM from part 1
- `stock_updown_model.pkl`: Decision Tree, scaler and feature names from part 2
- `stock_volatility_model.pkl`: Logistic Regression, scaler and feature names from part 3
- `app.py`: small Streamlit app that uses the saved models

## How to run

Open `SmartStock.ipynb` in Google Colab or Jupyter and run all the cells. The data downloads automatically, no CSV needed.

If you're running it locally:

```
pip install -r requirements.txt
```

## Streamlit app

`app.py` is a small web page on top of the saved models. You type a stock name (`.NS` for NSE and `.BO` for BSE) and it shows the closing price chart, the next day close from the LSTM, UP or DOWN from the Decision Tree, and BIG MOVE or QUIET from the Logistic Regression.

```
streamlit run app.py
```

It only loads the saved models, so run the notebook first if those files are missing. All the training is in the notebook.

## To do

- Try GRU and compare with SimpleRNN and LSTM
- Hyperparameter tuning with GridSearchCV
- Try the big move model on more stocks and check if it still beats the baselines
- Add news or events (results day, budget day) as features

This is just a learning project, not financial advice.
