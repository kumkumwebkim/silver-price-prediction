
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from prophet import Prophet
import xgboost as xgb
import os

def run_annual_forecasting():
    print("🚀 Starting Annual Silver Price Forecasting (Target: Dec 2026)")
    
    # 1. Data Collection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    print(f"📊 Fetching data from {start_date.date()} to {end_date.date()}")
    
    silver = yf.Ticker("SI=F")
    df = silver.history(start=start_date, end=end_date)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Save raw data
    df.to_csv('silver_prices_10y.csv', index=False)
    
    # 2. Feature Engineering
    print("🛠️ Engineering features...")
    df_ml = df.copy()
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        df_ml[f'Close_Lag_{lag}'] = df_ml['Close'].shift(lag)
    
    for window in [7, 14, 30, 60]:
        df_ml[f'Rolling_Mean_{window}'] = df_ml['Close'].rolling(window=window).mean()
        df_ml[f'Rolling_Std_{window}'] = df_ml['Close'].rolling(window=window).std()
    
    # RSI
    delta = df_ml['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_ml['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Time features
    df_ml['DayOfWeek'] = df_ml['Date'].dt.dayofweek
    df_ml['Month'] = df_ml['Date'].dt.month
    df_ml['Quarter'] = df_ml['Date'].dt.quarter
    
    df_ml_clean = df_ml.dropna()
    df_ml_clean.to_csv('silver_prices_featured.csv', index=False)
    
    # 3. Prophet Forecasting (Total 2026)
    print("🔮 Training Prophet model for 2026 forecast...")
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_df)
    
    # Forecast until Dec 31, 2026
    future_end = pd.Timestamp('2026-12-31')
    days_to_forecast = (future_end - prophet_df['ds'].max()).days
    future = model.make_future_dataframe(periods=days_to_forecast)
    forecast = model.predict(future)
    
    # 4. Results & Summary
    dec_2026_forecast = forecast[forecast['ds'].dt.month == 12]
    avg_dec_price = dec_2026_forecast['yhat'].mean()
    
    print("\n" + "="*40)
    print("🎯 ANNUAL FORECAST SUMMARY (DEC 2026)")
    print("="*40)
    print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")
    print(f"Projected Dec 2026 Avg: ${avg_dec_price:.2f}")
    print(f"Forecasted Range: ${dec_2026_forecast['yhat_lower'].min():.2f} - ${dec_2026_forecast['yhat_upper'].max():.2f}")
    print("="*40)

    # 4.1 Quarterly Analysis
    print("\n📊 QUARTERLY PERFORMANCE ANALYSIS")
    print("="*40)
    
    # Historical Quarterly Analysis
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    historical_quarterly = df.groupby(['Year', 'Quarter'])['Close'].agg(['first', 'last']).reset_index()
    historical_quarterly['Pct Change'] = ((historical_quarterly['last'] - historical_quarterly['first']) / historical_quarterly['first'] * 100).round(2)
    
    print("\n📈 Historical Quarterly Changes (Last 4 Quarters):")
    print(historical_quarterly.tail(4).to_string(index=False))

    # Forecasted Quarterly Analysis (2026)
    forecast_2026 = forecast[forecast['ds'].dt.year == 2026].copy()
    forecast_2026['Quarter'] = forecast_2026['ds'].dt.quarter
    forecast_quarterly = forecast_2026.groupby('Quarter')['yhat'].agg(['first', 'last']).reset_index()
    forecast_quarterly['Pct Change'] = ((forecast_quarterly['last'] - forecast_quarterly['first']) / forecast_quarterly['first'] * 100).round(2)
    
    print("\n🔮 Forecasted Quarterly Changes (2026):")
    forecast_quarterly['Quarter'] = '2026-Q' + forecast_quarterly['Quarter'].astype(str)
    print(forecast_quarterly.to_string(index=False))
    print("="*40)
    
    # 5. Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Historical'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='red')))
    fig.update_layout(title='Silver Price Forecast through 2026', xaxis_title='Date', yaxis_title='Price (USD)')
    
    # Save chart if possible, or just print success
    print("✅ Forecasting complete. Data saved to CSV files.")

if __name__ == "__main__":
    try:
        run_annual_forecasting()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("💡 Ensure yfinance, prophet, and plotly are installed.")
