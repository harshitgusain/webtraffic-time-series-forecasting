import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


st.set_page_config(page_title="Wikipedia Traffic Forecasting", layout="wide")
st.title("📊 Wikipedia Web Traffic Time Series Dashboard")

# ============================================
# 1️⃣ Preprocessing Pipeline with Caching
# ============================================
@st.cache_data
def load_and_preprocess(filepath: str):
    raw_df = pd.read_csv(filepath, nrows=1000)
    df = raw_df.melt(id_vars='Page', var_name='date', value_name='views')
    df['date'] = pd.to_datetime(df['date'])
    df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
    df = df.sort_values(by=['Page', 'date']).reset_index(drop=True)
    df = df[np.isfinite(df['views']) & (df['views'] >= 0)]
    
    # Filter pages based on data sufficiency
    page_stats = df.groupby('Page')['views'].agg(
        total_points='count',
        non_zero_points=lambda x: (x > 0).sum(),
        zero_ratio=lambda x: (x == 0).mean(),
        mean_views='mean'
    ).reset_index()
    
    filtered_pages = page_stats[
        (page_stats['total_points'] >= 500) &
        (page_stats['non_zero_points'] >= 300) &
        (page_stats['zero_ratio'] < 0.8)
    ]['Page']
    
    df = df[df['Page'].isin(filtered_pages)]
    
    # Regularize to daily frequency
    def regularize_page(page_df):
        page_df = page_df.set_index('date').asfreq('D')
        page_df['views'] = page_df['views'].fillna(0)
        page_df['Page'] = page_df['Page'].iloc[0]
        return page_df.reset_index()
    
    clean_df = df.groupby('Page', group_keys=False).apply(regularize_page)
    clean_df['log_views'] = np.log1p(clean_df['views'])
    clean_df.to_csv("clean1.csv")
    return clean_df

with st.spinner("🔄 Loading & cleaning dataset..."):
    df_long = load_and_preprocess("train_1.csv")

st.success(f"✅ Dataset ready: {df_long['Page'].nunique():,} pages, {len(df_long):,} records")

# ============================================
# 2️⃣ Sidebar Controls (Multi-Page Selection)
# ============================================
st.sidebar.header("🔍 Select Options")

pages = df_long['Page'].unique().tolist()
selected_pages = st.sidebar.multiselect(
    "Select Web Pages",
    options=pages,
    default=pages[:2] if len(pages) >= 2 else pages
)

view_type = st.sidebar.radio(
    "Select Value to Plot",
    options=["Raw Views", "Log Transformed"]
)

# ============================================
# 3️⃣ Visualization Section (Multiple Pages)
# ============================================
st.subheader("📈 Time Series Visualization")

fig, ax = plt.subplots(figsize=(14, 6))
for page in selected_pages:
    page_data = df_long[df_long['Page'] == page]
    y = page_data['views'] if view_type == "Raw Views" else page_data['log_views']
    ax.plot(page_data['date'], y, label=page)

ax.set_title(f"Time Series ({view_type})")
ax.set_xlabel("Date")
ax.set_ylabel("Views" if view_type == "Raw Views" else "log(views+1)")
ax.legend()
st.pyplot(fig)

# ============================================
# 4️⃣ Modeling Utilities
# ============================================
def train_test_split_series(series, test_size=30):
    return series[:-test_size], series[-test_size:]

def evaluate_forecast(true, predicted):
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mae = mean_absolute_error(true, predicted)
    mape = np.mean(np.abs((true - predicted) / true)) * 100
    return rmse, mae, mape

def forecast_model(model_name, train, test):
    if model_name == "ARIMA":
        model = ARIMA(train, order=(2,1,2))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))
    elif model_name == "SARIMA":
        model = SARIMAX(train, order=(2,1,2), seasonal_order=(1,1,1,7))
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=len(test))
    elif model_name == "Holt-Winters":
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7)
        fitted = model.fit()
        forecast = fitted.forecast(len(test))
    else:
        raise ValueError("Invalid model name")
    rmse, mae, mape = evaluate_forecast(test, forecast)
    return forecast, rmse, mae, mape, fitted

# ============================================
# 5️⃣ Forecasting Section (Single Page Forecast)
# ============================================
st.subheader("🔮 Forecasting & Model Analysis")

model_choice = st.selectbox("Choose Forecasting Model", ["ARIMA", "SARIMA", "Holt-Winters"])
test_size = st.slider("Test Data Size (days)", 30, 120, 60)
forecast_page = st.selectbox("Select Page for Forecasting", selected_pages)

if st.button("Run Forecast"):
    st.write(f"### Running {model_choice} model for: {forecast_page}")
    page_data = df_long[df_long['Page'] == forecast_page]
    ts = page_data.set_index('date')['log_views']
    ts = ts.asfreq('D').fillna(0)
    
    train, test = train_test_split_series(ts, test_size=test_size)
    
    try:
        forecast, rmse, mae, mape, fitted_model = forecast_model(model_choice, train, test)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train.index, train, label="Train", color='blue')
        ax.plot(test.index, test, label="Test", color='orange')
        ax.plot(test.index, forecast, label="Forecast", color='green')
        ax.set_title(f"{model_choice} Forecast — {forecast_page}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Views")
        ax.legend()
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("MAE", f"{mae:.2f}")
        col3.metric("MAPE (%)", f"{mape:.2f}")

    except Exception as e:
        st.error(f"❌ Model failed: {e}")

# ============================================
# 6️⃣ Optional Data View
# ============================================
with st.expander("📋 Show Cleaned Data for Selected Pages"):
    st.dataframe(df_long[df_long['Page'].isin(selected_pages)])
