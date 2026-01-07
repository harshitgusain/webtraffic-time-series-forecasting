import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

st.set_page_config(page_title="📊 Wikipedia Traffic Forecasting", layout="wide")
st.title("🌐 Wikipedia Web Traffic Forecasting Dashboard")

# ============================================
# 1️⃣ Data Loading
# ============================================
@st.cache_data
def load_and_preprocess(filepath: str):
    raw_df = pd.read_csv(filepath, nrows=1000)
    df = raw_df.melt(id_vars='Page', var_name='date', value_name='views')
    df['date'] = pd.to_datetime(df['date'])
    df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
    df = df.sort_values(by=['Page', 'date']).reset_index(drop=True)
    df = df[np.isfinite(df['views']) & (df['views'] >= 0)]
    
    # filter pages
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

    def regularize_page(page_df):
        page_df = page_df.set_index('date').asfreq('D')
        page_df['views'] = page_df['views'].fillna(0)
        page_df['Page'] = page_df['Page'].iloc[0]
        return page_df.reset_index()
    
    clean_df = df.groupby('Page', group_keys=False).apply(regularize_page)
    clean_df['log_views'] = np.log1p(clean_df['views'])
    return clean_df

with st.spinner("🔄 Loading & cleaning dataset..."):
    df_long = load_and_preprocess("train_1.csv")

st.success(f"✅ Data ready: {df_long['Page'].nunique():,} pages, {len(df_long):,} records")

# ============================================
# 2️⃣ Sidebar Options
# ============================================
st.sidebar.header("🔧 Configuration")

pages = df_long['Page'].unique().tolist()
selected_pages = st.sidebar.multiselect(
    "Select Pages", options=pages, default=pages[:2] if len(pages) >= 2 else pages
)

view_type = st.sidebar.radio("Value to Plot", ["Raw Views", "Log Transformed"])

test_size = st.sidebar.slider("Test Data Size (days)", 30, 120, 60)
model_choice = st.sidebar.selectbox("Model", ["ARIMA", "SARIMA", "Holt-Winters"])
forecast_page = st.sidebar.selectbox("Page for Forecast", selected_pages)

# ============================================
# 3️⃣ Visualization
# ============================================
tab1, tab2, tab3 = st.tabs(["📈 Visualization", "🧪 Diagnostics", "🔮 Forecasting"])

with tab1:
    st.subheader("Time Series Overview")
    fig, ax = plt.subplots(figsize=(14, 6))
    for page in selected_pages:
        page_data = df_long[df_long['Page'] == page]
        y = page_data['views'] if view_type == "Raw Views" else page_data['log_views']
        ax.plot(page_data['date'], y, label=page)
    ax.set_xlabel("Date")
    ax.set_ylabel("Views" if view_type == "Raw Views" else "log(Views+1)")
    ax.legend()
    st.pyplot(fig)

# ============================================
# 4️⃣ Utilities
# ============================================
def train_test_split_series(series, test_size=30):
    return series[:-test_size], series[-test_size:]

def evaluate_forecast(true, predicted):
    epsilon = 1e-5
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mae = mean_absolute_error(true, predicted)
    mape = np.mean(np.abs((true - predicted) / np.maximum(true, epsilon))) * 100
    wmape = np.sum(np.abs(true - predicted)) / np.sum(np.abs(true) + epsilon) * 100
    return rmse, mae, mape, wmape

# ============================================
# 5️⃣ Diagnostics & Forecasting
# ============================================
with tab2:
    st.subheader(f"Diagnostics for {forecast_page}")
    page_data = df_long[df_long['Page'] == forecast_page]
    ts = page_data.set_index('date')['log_views'].asfreq('D').fillna(0)
    train, test = train_test_split_series(ts, test_size=test_size)

    # Stationarity tests
    adf_stat, adf_p = adfuller(train)[0], adfuller(train)[1]
    kpss_stat, kpss_p = kpss(train, regression='c', nlags="auto")[0], kpss(train, regression='c', nlags="auto")[1]
    
    st.write(f"**ADF test** — Statistic: `{adf_stat:.4f}`, p-value: `{adf_p:.4f}`")
    st.write(f"**KPSS test** — Statistic: `{kpss_stat:.4f}`, p-value: `{kpss_p:.4f}`")

    # ACF/PACF plots
    st.write("Autocorrelation & Partial Autocorrelation:")
    fig_acf, ax_acf = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(train.dropna(), lags=40, ax=ax_acf[0])
    ax_acf[0].set_title("ACF (log series)")
    plot_pacf(train.dropna(), lags=40, ax=ax_acf[1], method='ywm')
    ax_acf[1].set_title("PACF (log series)")
    st.pyplot(fig_acf)

# ============================================
# 6️⃣ Forecasting
# ============================================
with tab3:
    st.subheader(f"Forecasting using {model_choice} for {forecast_page}")
    if st.button("Run Forecast"):
        ts = page_data.set_index('date')['log_views'].asfreq('D').fillna(0)
        train, test = train_test_split_series(ts, test_size=test_size)

        try:
            if model_choice == "ARIMA":
                model = ARIMA(train, order=(2,1,2))
                fitted = model.fit()
                forecast_log = fitted.forecast(steps=len(test))
            elif model_choice == "SARIMA":
                model = SARIMAX(train, order=(2,1,2), seasonal_order=(1,1,1,7))
                fitted = model.fit(disp=False)
                forecast_log = fitted.forecast(steps=len(test))
            else:  # Holt-Winters
                ts_raw = page_data.set_index('date')['views'].asfreq('D').fillna(0)
                train_raw, test_raw = train_test_split_series(ts_raw, test_size=test_size)
                model = ExponentialSmoothing(train_raw, trend='add', seasonal='add', seasonal_periods=7)
                fitted = model.fit()
                forecast_raw = fitted.forecast(len(test_raw))
                forecast_log = None

            if model_choice in ["ARIMA", "SARIMA"]:
                forecast_orig = np.expm1(forecast_log)
                actual_orig = np.expm1(test)
                train_orig = np.expm1(train)
            else:
                forecast_orig = forecast_raw
                actual_orig = test_raw
                train_orig = train_raw

            # Plot
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(train_orig.index, train_orig, label="Train", color='blue')
            ax2.plot(actual_orig.index, actual_orig, label="Test", color='orange')
            ax2.plot(actual_orig.index, forecast_orig, label="Forecast", color='green')
            ax2.legend()
            ax2.set_title(f"{model_choice} Forecast on Original Scale")
            st.pyplot(fig2)

            # Metrics
            rmse, mae, mape, wmape = evaluate_forecast(actual_orig, forecast_orig)
            st.markdown("### 📊 Model Evaluation Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("MAE", f"{mae:.2f}")
            col3.metric("MAPE", f"{mape:.2f}%")
            col4.metric("WMAPE", f"{wmape:.2f}%")

            # Visual gauge for MAPE quality
            if wmape < 10:
                color = "🟢 Excellent"
            elif wmape < 30:
                color = "🟡 Acceptable"
            else:
                color = "🔴 Poor"
            st.markdown(f"**Overall Forecast Accuracy:** {color}")

        except Exception as e:
            st.error(f"❌ Forecast failed: {e}")

# ============================================
# 7️⃣ Optional data view
# ============================================
with st.expander("📋 View Cleaned Data"):
    st.dataframe(df_long[df_long['Page'].isin(selected_pages)])
