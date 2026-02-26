"""
Silver Price Prediction Streamlit Application
A comprehensive dashboard for silver price analysis and prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# USD to INR conversion rate
USD_TO_INR = 83.50  # Approximate exchange rate

def format_inr(amount):
    """Format amount in Indian Rupees with ₹ symbol"""
    if amount >= 10000000:  # 1 Crore
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"₹{amount/100000:.2f} L"
    else:
        return f"₹{amount:,.2f}"

# Page configuration
st.set_page_config(
    page_title="🪙 Silver Price Prediction",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #C0C0C0;
        --secondary-color: #1E3A5F;
        --accent-color: #4ECDC4;
        --background-dark: #0E1117;
        --text-light: #FAFAFA;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2D5A87 50%, #3D7AB5 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: #C0C0C0;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(192, 192, 192, 0.2);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #C0C0C0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.5rem;
    }
    
    .positive { color: #00D26A !important; }
    .negative { color: #FF4757 !important; }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E3A5F 0%, #0E1117 100%);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1E3A5F, transparent);
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        border-left: 4px solid #C0C0C0;
    }
    
    /* Chat styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #1E3A5F, #2D5A87);
        margin-left: 20%;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #16213e, #1a1a2e);
        margin-right: 20%;
        border-left: 3px solid #C0C0C0;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #C0C0C0 0%, #A0A0A0 100%);
        color: #1E3A5F;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(192, 192, 192, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("silver_prices_featured.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h2 style="color: #C0C0C0;">🪙 Silver Analytics</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Main", "📊 EDA", "📈 Monitoring Reports", "📏 Performance Measures", 
     "🔮 Predictions", "📋 Data Report", "💬 AI Chat"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 0.8rem; color: #888;">
    <p>Silver Price Analysis Dashboard</p>
    <p>Data: 2016 - 2026</p>
</div>
""", unsafe_allow_html=True)

# ============== PAGE: MAIN ==============
if page == "🏠 Main":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🪙 Silver Price Prediction Dashboard</h1>
        <p style="color: #AAA; margin-top: 0.5rem;">Comprehensive Analysis & Forecasting Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    change = latest['Close'] - prev['Close']
    change_pct = (change / prev['Close']) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"₹{latest['Close'] * USD_TO_INR:,.2f}",
            delta=f"₹{change * USD_TO_INR:,.2f} ({change_pct:.2f}%)"
        )
    
    with col2:
        st.metric(
            label="52-Week High",
            value=f"₹{df['High'].tail(252).max() * USD_TO_INR:,.2f}",
            delta=f"{((latest['Close'] / df['High'].tail(252).max()) - 1) * 100:.1f}% from high"
        )
    
    with col3:
        st.metric(
            label="52-Week Low",
            value=f"₹{df['Low'].tail(252).min() * USD_TO_INR:,.2f}",
            delta=f"{((latest['Close'] / df['Low'].tail(252).min()) - 1) * 100:.1f}% from low"
        )
    
    with col4:
        st.metric(
            label="Avg Volume",
            value=f"{df['Volume'].tail(30).mean():,.0f}",
            delta="30-day average"
        )
    
    st.markdown("---")
    
    # Price Chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Price History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#C0C0C0', width=2),
            fill='tozeroy',
            fillcolor='rgba(192, 192, 192, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA_30'],
            mode='lines',
            name='30-Day MA',
            line=dict(color='#4ECDC4', width=1.5, dash='dash')
        ))
        fig.update_layout(
            template='plotly_dark',
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            xaxis_title="",
            yaxis_title="Price (₹)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        stats_df = pd.DataFrame({
            'Metric': ['Mean Price', 'Std Dev', 'Min Price', 'Max Price', 
                      'Latest RSI', 'Volatility (20d)'],
            'Value': [
                f"₹{df['Close'].mean() * USD_TO_INR:,.2f}",
                f"₹{df['Close'].std() * USD_TO_INR:,.2f}",
                f"₹{df['Close'].min() * USD_TO_INR:,.2f}",
                f"₹{df['Close'].max() * USD_TO_INR:,.2f}",
                f"{latest['RSI_14']:.1f}",
                f"{latest['Volatility_20']:.1f}%"
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        # Mini trend indicator
        st.markdown("### 📍 Trend Signals")
        ma_signal = "🟢 Bullish" if latest['Close'] > latest['MA_30'] else "🔴 Bearish"
        rsi_signal = "🟡 Overbought" if latest['RSI_14'] > 70 else ("🟡 Oversold" if latest['RSI_14'] < 30 else "🟢 Neutral")
        
        st.markdown(f"**MA Signal:** {ma_signal}")
        st.markdown(f"**RSI Signal:** {rsi_signal}")

# ============== PAGE: EDA ==============
elif page == "📊 EDA":
    st.markdown("## 📊 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Analysis", "📊 Distributions", "🔗 Correlations", "📅 Seasonality"])
    
    with tab1:
        st.markdown("### Candlestick Chart with Moving Averages")
        
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", df['Date'].min())
        with col2:
            end_date = st.date_input("End Date", df['Date'].max())
        
        mask = (df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))
        filtered_df = df[mask]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(
            x=filtered_df['Date'],
            open=filtered_df['Open'] * USD_TO_INR, 
            high=filtered_df['High'] * USD_TO_INR,
            low=filtered_df['Low'] * USD_TO_INR, 
            close=filtered_df['Close'] * USD_TO_INR,
            name='OHLC'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA_7'] * USD_TO_INR,
                                name='7-Day MA', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA_30'] * USD_TO_INR,
                                name='30-Day MA', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA_90'] * USD_TO_INR,
                                name='90-Day MA', line=dict(color='green', width=1)), row=1, col=1)
        
        fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Volume'],
                            name='Volume', marker_color='rgba(192, 192, 192, 0.5)'), row=2, col=1)
        
        fig.update_layout(template='plotly_dark', height=600, showlegend=True,
                         xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Price Distribution")
            df_inr = df.copy()
            df_inr['Close_INR'] = df_inr['Close'] * USD_TO_INR
            fig = px.histogram(df_inr, x='Close_INR', nbins=50, 
                             color_discrete_sequence=['#C0C0C0'],
                             labels={'Close_INR': 'Price (₹)'})
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Returns Distribution")
            fig = px.histogram(df, x='Returns', nbins=50,
                             color_discrete_sequence=['#4ECDC4'])
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Box Plot by Year")
        df_inr['Year'] = df_inr['Date'].dt.year
        fig = px.box(df_inr, x='Year', y='Close_INR', 
                    color_discrete_sequence=['#C0C0C0'],
                    labels={'Close_INR': 'Price (₹)'})
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Correlation Heatmap")
        corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                    'MA_7', 'MA_30', 'RSI_14', 'Volatility_20']
        corr_matrix = df[corr_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        fig.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Monthly Average Prices")
        df_inr = df.copy()
        df_inr['Close_INR'] = df_inr['Close'] * USD_TO_INR
        monthly_avg = df_inr.groupby('Month')['Close_INR'].mean().reset_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_avg['MonthName'] = monthly_avg['Month'].apply(lambda x: month_names[int(x)-1])
        
        fig = px.bar(monthly_avg, x='MonthName', y='Close_INR',
                    color='Close_INR', color_continuous_scale='Blues',
                    labels={'Close_INR': 'Avg Price (₹)'})
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Yearly Trend")
        yearly_avg = df_inr.groupby('Year')['Close_INR'].mean().reset_index()
        fig = px.line(yearly_avg, x='Year', y='Close_INR', markers=True,
                     labels={'Close_INR': 'Avg Price (₹)'})
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============== PAGE: MONITORING REPORTS ==============
elif page == "📈 Monitoring Reports":
    st.markdown("## 📈 Monitoring Reports")
    
    tab1, tab2, tab3 = st.tabs(["📊 Technical Indicators", "🔔 Alerts", "📉 Volatility"])
    
    with tab1:
        st.markdown("### Technical Indicators Dashboard")
        
        # RSI Chart
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=('Price with Bollinger Bands', 'RSI (14)', 'MACD'))
        
        # Price with Bollinger Bands
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price',
                                line=dict(color='#C0C0C0')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper',
                                line=dict(color='red', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower',
                                line=dict(color='green', dash='dash'), fill='tonexty',
                                fillcolor='rgba(100, 100, 100, 0.1)'), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], name='RSI',
                                line=dict(color='#4ECDC4')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD',
                                line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal',
                                line=dict(color='orange')), row=3, col=1)
        
        fig.update_layout(template='plotly_dark', height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔔 Price Alert Thresholds")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            upper_threshold = st.number_input("Upper Alert (₹)", value=35.0 * USD_TO_INR, step=50.0)
        with col2:
            lower_threshold = st.number_input("Lower Alert (₹)", value=20.0 * USD_TO_INR, step=50.0)
        with col3:
            current_price = df['Close'].iloc[-1]
            st.metric("Current Price", f"₹{current_price * USD_TO_INR:,.2f}")
        
        # Alert status
        if (current_price * USD_TO_INR) >= upper_threshold:
            st.error(f"⚠️ ALERT: Price ₹{current_price * USD_TO_INR:,.2f} is ABOVE upper threshold ₹{upper_threshold:,.2f}")
        elif (current_price * USD_TO_INR) <= lower_threshold:
            st.warning(f"⚠️ ALERT: Price ₹{current_price * USD_TO_INR:,.2f} is BELOW lower threshold ₹{lower_threshold:,.2f}")
        else:
            st.success(f"✅ Price ₹{current_price * USD_TO_INR:,.2f} is within normal range")
        
        # Historical alerts chart
        st.markdown("### Historical Price vs Thresholds")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price',
                                line=dict(color='#C0C0C0')))
        fig.add_hline(y=upper_threshold, line_dash="dash", line_color="red", 
                     annotation_text="Upper Alert")
        fig.add_hline(y=lower_threshold, line_dash="dash", line_color="green",
                     annotation_text="Lower Alert")
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📉 Volatility Analysis")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=('Price', 'Annualized Volatility (%)'))
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price',
                                line=dict(color='#C0C0C0')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Volatility_20'], name='20-Day Vol',
                                line=dict(color='#4ECDC4')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Volatility_60'], name='60-Day Vol',
                                line=dict(color='orange')), row=2, col=1)
        
        fig.update_layout(template='plotly_dark', height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Volatility stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current 20-Day Vol", f"{df['Volatility_20'].iloc[-1]:.1f}%")
        with col2:
            st.metric("Avg 20-Day Vol", f"{df['Volatility_20'].mean():.1f}%")
        with col3:
            st.metric("Max 20-Day Vol", f"{df['Volatility_20'].max():.1f}%")

# ============== PAGE: PERFORMANCE MEASURES ==============
elif page == "📏 Performance Measures":
    st.markdown("## 📏 Model Performance Measures")
    
    # Prepare data for modeling
    feature_cols = ['MA_7', 'MA_30', 'MA_90', 'RSI_14', 'MACD', 'Volatility_20',
                   'Close_Lag_1', 'Close_Lag_7', 'Momentum_7']
    
    model_df = df.dropna(subset=feature_cols + ['Close']).copy()
    X = model_df[feature_cols]
    y = model_df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results.append({
            'Model': name,
            'MAE (₹)': f"₹{mae * USD_TO_INR:.4f}",
            'RMSE (₹)': f"₹{rmse * USD_TO_INR:.4f}",
            'MAPE': f"{mape:.2f}%",
            'R² Score': f"{r2:.4f}"
        })
    
    # Display results
    st.markdown("### 📊 Model Comparison")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, hide_index=True, use_container_width=True)
    
    # Actual vs Predicted Chart
    st.markdown("### 📈 Actual vs Predicted (Test Set)")
    
    selected_model = st.selectbox("Select Model", list(models.keys()))
    
    fig = go.Figure()
    test_dates = model_df['Date'].iloc[-len(y_test):]
    
    fig.add_trace(go.Scatter(x=test_dates, y=y_test.values, name='Actual',
                            line=dict(color='#C0C0C0', width=2)))
    fig.add_trace(go.Scatter(x=test_dates, y=predictions[selected_model], name='Predicted',
                            line=dict(color='#4ECDC4', width=2, dash='dash')))
    
    fig.update_layout(template='plotly_dark', height=400,
                     xaxis_title='Date', yaxis_title='Price (₹)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (for RF)
    st.markdown("### 🎯 Feature Importance (Random Forest)")
    rf_model = models['Random Forest']
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Blues')
    fig.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============== PAGE: PREDICTIONS ==============
elif page == "🔮 Predictions":
    st.markdown("## 🔮 Price Predictions")
    
    # Model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model_type = st.selectbox("Select Prediction Model", 
                                  ["Gradient Boosting", "Random Forest", "Linear Regression"])
        forecast_days = st.slider("Forecast Days", 7, 90, 30)
    
    # Prepare and train model
    feature_cols = ['MA_7', 'MA_30', 'RSI_14', 'MACD', 'Volatility_20',
                   'Close_Lag_1', 'Close_Lag_7', 'Momentum_7']
    
    model_df = df.dropna(subset=feature_cols + ['Close']).copy()
    X = model_df[feature_cols]
    y = model_df['Close']
    
    if model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = LinearRegression()
    
    model.fit(X, y)
    
    # Generate future predictions (simplified approach)
    last_row = model_df.iloc[-1:][feature_cols]
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), 
                                 periods=forecast_days, freq='D')
    
    predictions = []
    current_features = last_row.copy()
    last_price = df['Close'].iloc[-1]
    
    for i in range(forecast_days):
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        
        # Update features for next prediction (simplified)
        current_features['Close_Lag_1'] = pred
        if i >= 6:
            current_features['Close_Lag_7'] = predictions[i-6]
        current_features['MA_7'] = np.mean(predictions[-7:]) if len(predictions) >= 7 else pred
        last_price = pred
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions
    })
    
    with col2:
        st.markdown("### Forecast Summary")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Last Actual Price", f"₹{df['Close'].iloc[-1] * USD_TO_INR:,.2f}")
        with col_b:
            st.metric(f"Predicted ({forecast_days}d)", f"₹{predictions[-1] * USD_TO_INR:,.2f}",
                     delta=f"{((predictions[-1] / df['Close'].iloc[-1]) - 1) * 100:.1f}%")
        with col_c:
            st.metric("Trend", "📈 Up" if predictions[-1] > df['Close'].iloc[-1] else "📉 Down")
    
    # Forecast chart
    st.markdown("### 📈 Price Forecast")
    fig = go.Figure()
    
    # Historical data (last 90 days)
    historical = df.tail(90)
    fig.add_trace(go.Scatter(x=historical['Date'], y=historical['Close'],
                            name='Historical', line=dict(color='#C0C0C0', width=2)))
    
    # Predictions
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted_Price'],
                            name='Forecast', line=dict(color='#4ECDC4', width=2, dash='dash')))
    
    # Confidence interval (simplified)
    upper = forecast_df['Predicted_Price'] * 1.05
    lower = forecast_df['Predicted_Price'] * 0.95
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=upper, name='Upper Bound',
                            line=dict(color='rgba(78, 205, 196, 0.3)'), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=lower, name='Lower Bound',
                            fill='tonexty', fillcolor='rgba(78, 205, 196, 0.1)',
                            line=dict(color='rgba(78, 205, 196, 0.3)'), showlegend=False))
    
    fig.update_layout(template='plotly_dark', height=500,
                     xaxis_title='Date', yaxis_title='Price (₹)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Download predictions
    st.markdown("### 📥 Download Predictions")
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="silver_price_predictions.csv",
        mime="text/csv"
    )
    
    forecast_inr_df = forecast_df.copy()
    forecast_inr_df['Predicted_Price_INR'] = (forecast_inr_df['Predicted_Price'] * USD_TO_INR).round(2)
    st.dataframe(forecast_inr_df[['Date', 'Predicted_Price_INR']].head(10), hide_index=True, use_container_width=True)

# ============== PAGE: DATA REPORT ==============
elif page == "📋 Data Report":
    st.markdown("## 📋 Data Report")
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Statistics", "🔍 Quality"])
    
    with tab1:
        st.markdown("### Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", f"{len(df.columns)}")
        with col3:
            st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
        with col4:
            st.metric("Missing Values", f"{df.isnull().sum().sum()}")
        
        st.markdown("### Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null': df.notna().sum().values,
            'Null': df.isnull().sum().values
        })
        st.dataframe(col_info, hide_index=True, use_container_width=True, height=400)
    
    with tab2:
        st.markdown("### Descriptive Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df[numeric_cols].describe().round(4), use_container_width=True)
        
        st.markdown("### Value Distributions")
        selected_col = st.selectbox("Select Column", numeric_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=selected_col, nbins=50,
                             color_discrete_sequence=['#C0C0C0'])
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=selected_col, color_discrete_sequence=['#4ECDC4'])
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Data Quality Report")
        
        # Calculate quality metrics
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        quality_score = ((total_cells - missing_cells) / total_cells) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Quality Score", f"{quality_score:.1f}%")
        with col2:
            st.metric("Complete Rows", f"{len(df.dropna()):,}")
        with col3:
            st.metric("Duplicate Rows", f"{df.duplicated().sum()}")
        
        st.markdown("### Missing Value Analysis")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum().values,
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2).values
        }).sort_values('Missing', ascending=False)
        
        missing_with_nulls = missing_df[missing_df['Missing'] > 0]
        if len(missing_with_nulls) > 0:
            fig = px.bar(missing_with_nulls.head(20), x='Column', y='Percentage',
                        color='Percentage', color_continuous_scale='Reds')
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values in the dataset!")
        
        # Export data
        st.markdown("### 📥 Export Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset",
            data=csv,
            file_name="silver_prices_full.csv",
            mime="text/csv"
        )

# ============== PAGE: AI CHAT ==============
elif page == "💬 AI Chat":
    st.markdown("## 💬 Ask Questions About the Data")
    st.markdown("*Ask me anything about silver prices and I'll analyze the data for you!*")
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        st.markdown("""
        - What was the highest silver price?
        - What is the average price in 2024?
        - Show me the price trend
        - What is the current volatility?
        - When did silver reach its peak?
        - Compare 2023 vs 2024 prices
        - What is the correlation between price and volume?
        """)
    
    # Chat input
    user_question = st.text_input("Ask a question about silver prices:", 
                                   placeholder="e.g., What was the highest price in 2024?")
    
    if user_question:
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Process the question (rule-based NLU)
        response = ""
        show_chart = False
        chart_data = None
        
        question_lower = user_question.lower()
        
        # Highest/Maximum price queries
        if any(word in question_lower for word in ['highest', 'maximum', 'max', 'peak', 'top']):
            if '2024' in question_lower:
                data_2024 = df[df['Year'] == 2024]
                if len(data_2024) > 0:
                    max_price = data_2024['Close'].max()
                    max_date = data_2024.loc[data_2024['Close'].idxmax(), 'Date']
                    response = f"📈 The highest silver price in 2024 was **₹{max_price * USD_TO_INR:,.2f}** on **{max_date.strftime('%B %d, %Y')}**."
            elif '2023' in question_lower:
                data_2023 = df[df['Year'] == 2023]
                if len(data_2023) > 0:
                    max_price = data_2023['Close'].max()
                    max_date = data_2023.loc[data_2023['Close'].idxmax(), 'Date']
                    response = f"📈 The highest silver price in 2023 was **₹{max_price * USD_TO_INR:,.2f}** on **{max_date.strftime('%B %d, %Y')}**."
            else:
                max_price = df['Close'].max()
                max_date = df.loc[df['Close'].idxmax(), 'Date']
                response = f"📈 The all-time highest silver price in our dataset was **₹{max_price * USD_TO_INR:,.2f}** on **{max_date.strftime('%B %d, %Y')}**."
        
        # Lowest/Minimum price queries
        elif any(word in question_lower for word in ['lowest', 'minimum', 'min', 'bottom']):
            min_price = df['Close'].min()
            min_date = df.loc[df['Close'].idxmin(), 'Date']
            response = f"📉 The lowest silver price was **₹{min_price * USD_TO_INR:,.2f}** on **{min_date.strftime('%B %d, %Y')}**."
        
        # Average price queries
        elif any(word in question_lower for word in ['average', 'mean', 'avg']):
            if '2024' in question_lower:
                avg = df[df['Year'] == 2024]['Close'].mean()
                response = f"📊 The average silver price in 2024 was **₹{avg * USD_TO_INR:,.2f}**."
            elif '2023' in question_lower:
                avg = df[df['Year'] == 2023]['Close'].mean()
                response = f"📊 The average silver price in 2023 was **₹{avg * USD_TO_INR:,.2f}**."
            else:
                avg = df['Close'].mean()
                response = f"📊 The overall average silver price is **₹{avg * USD_TO_INR:,.2f}**."
        
        # Trend queries
        elif any(word in question_lower for word in ['trend', 'chart', 'graph', 'show']):
            response = "📈 Here's the silver price trend:"
            show_chart = True
            chart_data = df[['Date', 'Close', 'MA_30']].tail(252)
        
        # Volatility queries
        elif 'volatility' in question_lower:
            current_vol = df['Volatility_20'].iloc[-1]
            avg_vol = df['Volatility_20'].mean()
            response = f"📊 Current 20-day volatility is **{current_vol:.1f}%** (Average: {avg_vol:.1f}%)."
        
        # Current price queries
        elif any(word in question_lower for word in ['current', 'now', 'today', 'latest']):
            latest = df.iloc[-1]
            response = f"💰 The latest silver price is **₹{latest['Close'] * USD_TO_INR:,.2f}** as of {latest['Date'].strftime('%B %d, %Y')}."
        
        # Correlation queries
        elif 'correlation' in question_lower:
            if 'volume' in question_lower:
                corr = df['Close'].corr(df['Volume'])
                response = f"📊 The correlation between price and volume is **{corr:.4f}**."
            else:
                response = "📊 Price is strongly correlated with Open, High, Low (>0.99) and moderately with MA indicators."
        
        # Compare queries
        elif 'compare' in question_lower or 'vs' in question_lower:
            if '2023' in question_lower and '2024' in question_lower:
                avg_2023 = df[df['Year'] == 2023]['Close'].mean()
                avg_2024 = df[df['Year'] == 2024]['Close'].mean()
                change = ((avg_2024 - avg_2023) / avg_2023) * 100
                response = f"📊 **2023 vs 2024 Comparison:**\n- 2023 Avg: ₹{avg_2023 * USD_TO_INR:,.2f}\n- 2024 Avg: ₹{avg_2024 * USD_TO_INR:,.2f}\n- Change: {change:+.1f}%"
        
        # RSI queries
        elif 'rsi' in question_lower:
            current_rsi = df['RSI_14'].iloc[-1]
            signal = "overbought (>70)" if current_rsi > 70 else ("oversold (<30)" if current_rsi < 30 else "neutral")
            response = f"📊 Current RSI (14) is **{current_rsi:.1f}** - Market is {signal}."
        
        # Default response
        else:
            response = f"""🤔 I understand you want to know about: *{user_question}*

Here are some quick stats:
- **Current Price:** ₹{df['Close'].iloc[-1] * USD_TO_INR:,.2f}
- **52-Week Range:** ₹{df['Close'].tail(252).min() * USD_TO_INR:,.2f} - ₹{df['Close'].tail(252).max() * USD_TO_INR:,.2f}
- **Average Price:** ₹{df['Close'].mean() * USD_TO_INR:,.2f}

Try asking specific questions like "What was the highest price?" or "Show me the trend"."""
        
        # Add response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display chat history
        for msg in st.session_state.chat_history[-6:]:  # Show last 6 messages
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Show chart if requested
        if show_chart and chart_data is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['Close'],
                                    name='Price', line=dict(color='#C0C0C0')))
            fig.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['MA_30'],
                                    name='30-Day MA', line=dict(color='#4ECDC4', dash='dash')))
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>🪙 Silver Price Prediction Dashboard | Built with Streamlit</p>
    <p>Data Source: Yahoo Finance | Last Updated: January 2026</p>
</div>
""", unsafe_allow_html=True)
