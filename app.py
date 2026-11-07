import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

st.title("SmartStock")
st.write("Next day stock prediction using LSTM(price) and Decision Tree(up or down)")

#loading the saved models from the notebook
lstm_model=load_model("stock_lstm_model.keras")
with open("stock_updown_model.pkl","rb") as f:
  model_data=pickle.load(f)
updown_model=model_data["model"]
scaler=model_data["scaler"]
feature_names=model_data["features_names"]
with open("stock_volatility_model.pkl","rb") as f:
  vol_data=pickle.load(f)
vol_model=vol_data["model"]
vol_scaler=vol_data["scaler"]
vol_features=vol_data["features_names"]

stock_name=st.text_input("Enter Stock Name(.NS for NSE and .BO for BSE)","LT.NS")

if st.button("Predict"):
  df=yf.download(stock_name,start=dt.datetime(2000,1,1),end=dt.datetime.now())
  if len(df)<200:
    st.error("Stock not found or not enough data")
    st.stop()
  df.columns=df.columns.droplevel(1)
  df=df.dropna().reset_index()

  #closing price with moving averages
  df["MA100"]=df["Close"].rolling(100).mean()
  df["MA200"]=df["Close"].rolling(200).mean()

  st.subheader("Closing Price")
  fig=plt.figure(figsize=(12,6))
  plt.plot(df["Date"],df["Close"],label="Close Price",linewidth=1)
  plt.plot(df["Date"],df["MA100"],label="MA 100",linewidth=1)
  plt.plot(df["Date"],df["MA200"],label="MA 200",linewidth=1)
  plt.legend()
  st.pyplot(fig)

  #next day price using lstm(same as part 1 of SmartStock.ipynb)
  last_100_days=df["Close"].values[-100:]
  last_price=last_100_days[-1]
  X_input=last_100_days/last_price
  X_input=X_input.reshape(1,100,1)
  next_day_price=lstm_model.predict(X_input)[0][0]*last_price

  #up or down using decision tree(same features as part 2 of SmartStock.ipynb)
  nifty=yf.download("^NSEI",start=dt.datetime.now()-dt.timedelta(days=200),end=dt.datetime.now())
  nifty.columns=nifty.columns.droplevel(1)
  nifty=nifty.dropna().reset_index()
  nifty=nifty[["Date","Close"]]
  nifty.columns=["Date","Nifty_Close"]

  new_df=pd.merge(df,nifty,on="Date",how="inner")
  new_df["Return"]=new_df["Close"].pct_change()
  new_df["Nifty_Return"]=new_df["Nifty_Close"].pct_change()
  new_df["Return_5"]=new_df["Close"].pct_change(5)
  new_df["Volume_Change"]=new_df["Volume"].pct_change()
  new_df["Day_Range"]=(new_df["High"]-new_df["Low"])/new_df["Close"]
  new_df["MA20"]=new_df["Close"].rolling(20).mean()
  new_df["MA50"]=new_df["Close"].rolling(50).mean()
  new_df["Close_MA20"]=new_df["Close"]/new_df["MA20"]
  new_df["Close_MA50"]=new_df["Close"]/new_df["MA50"]

  input_data=new_df[feature_names].tail(1)
  input_scaled=scaler.transform(input_data)
  prediction=updown_model.predict(input_scaled)
  pred_prob=updown_model.predict_proba(input_scaled)

  #big move or quiet using logistic regression(same features as part 3 of SmartStock.ipynb)
  new_df["Range_5"]=new_df["Day_Range"].rolling(5).mean()
  new_df["Range_20"]=new_df["Day_Range"].rolling(20).mean()
  new_df["Volatility_5"]=new_df["Return"].rolling(5).std()
  new_df["Volatility_20"]=new_df["Return"].rolling(20).std()
  new_df["Nifty_Volatility_20"]=new_df["Nifty_Return"].rolling(20).std()
  new_df["Volume_20"]=new_df["Volume"].rolling(20).mean()
  new_df["Range_Ratio"]=new_df["Day_Range"]/new_df["Range_20"]
  new_df["Range5_Ratio"]=new_df["Range_5"]/new_df["Range_20"]
  new_df["Move_Ratio"]=new_df["Return"].abs()/new_df["Volatility_20"]
  new_df["Volatility_Ratio"]=new_df["Volatility_5"]/new_df["Volatility_20"]
  new_df["Nifty_Ratio"]=new_df["Nifty_Return"].abs()/new_df["Nifty_Volatility_20"]
  new_df["Volume_Ratio"]=new_df["Volume"]/new_df["Volume_20"]

  vol_input=vol_scaler.transform(new_df[vol_features].tail(1))
  vol_prediction=vol_model.predict(vol_input)
  vol_prob=vol_model.predict_proba(vol_input)

  st.subheader("Prediction")
  col1,col2=st.columns(2)
  col1.metric("Last Close",f"{last_price:.2f}")
  col2.metric("Predicted Next Close(LSTM)",f"{next_day_price:.2f}")
  col3,col4=st.columns(2)
  col3.metric("Next Day(Decision Tree)","UP" if prediction[0]==1 else "DOWN",f"{pred_prob.max()*100:.0f}% sure",delta_color="off")
  col4.metric("Next Day Move(Logistic Regression)","BIG MOVE" if vol_prediction[0]==1 else "QUIET",f"{vol_prob.max()*100:.0f}% sure",delta_color="off")

  st.caption("In testing the price and up/down models were only around the baseline(LSTM RMSE close to yesterday price, Decision Tree around 50%). The big move/quiet model was the only one that beat its baselines(63% vs 59%). This is a learning project, not financial advice.")

  st.subheader("Last 10 Days Data")
  st.dataframe(df[["Date","Open","High","Low","Close","Volume"]].tail(10))
