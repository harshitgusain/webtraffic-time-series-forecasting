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
    ts = page_data.set_index('date')['log_views']   # already log-transformed
    ts = ts.asfreq('D').fillna(0)

    # ----- SPLIT -----
    train, test = train_test_split_series(ts, test_size=test_size)

    # ----- STATIONARITY TESTS -----
    st.subheader("🧪 Stationarity Tests (on training data)")
    # ADF
    try:
        adf_res = adfuller(train.dropna(), autolag='AIC')
        adf_stat, adf_p = adf_res[0], adf_res[1]
        st.write(f"**ADF test** — Statistic: `{adf_stat:.4f}`, p-value: `{adf_p:.4f}`")
        if adf_p < 0.05:
            st.write("→ ADF indicates the series is likely *stationary* (reject H0).")
        else:
            st.write("→ ADF indicates the series is likely *non-stationary* (fail to reject H0).")
    except Exception as e:
        st.write("ADF test failed:", e)
        adf_p = None

    # KPSS
    try:
        kpss_res = kpss(train.dropna(), regression='c', nlags="auto")
        kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
        st.write(f"**KPSS test** — Statistic: `{kpss_stat:.4f}`, p-value: `{kpss_p:.4f}`")
        if kpss_p < 0.05:
            st.write("→ KPSS indicates the series is likely *non-stationary* (reject H0).")
        else:
            st.write("→ KPSS indicates the series is likely *stationary* (fail to reject H0).")
    except Exception as e:
        st.write("KPSS test failed:", e)
        kpss_p = None

    # ----- ACF / PACF PLOTS -----
    st.subheader("📈 ACF & PACF (training series)")
    fig_acf, ax_acf = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(train.dropna(), lags=40, ax=ax_acf[0], alpha=0.05)
    ax_acf[0].set_title("ACF (log series)")
    plot_pacf(train.dropna(), lags=40, ax=ax_acf[1], alpha=0.05, method='ywm')
    ax_acf[1].set_title("PACF (log series)")
    st.pyplot(fig_acf)

    # ----- SIMPLE HEURISTIC FOR ORDER SELECTION -----
    st.subheader("🧭 Suggested Orders (heuristic)")
    # determine d from stationarity tests: if either ADF fails to reject OR KPSS rejects => difference
    d = 0
    try:
        if (adf_p is None or adf_p >= 0.05) or (kpss_p is not None and kpss_p < 0.05):
            d = 1
    except:
        d = 1

    # significance threshold for ACF/PACF
    n = len(train.dropna())
    thresh = 1.96 / np.sqrt(n) if n>0 else 0.0

    # compute sample acf/pacf values (using pandas autocorr for a few lags)
    max_lag = 10
    acf_vals = [train.dropna().autocorr(lag=lag) for lag in range(1, max_lag+1)]
    pacf_vals = []
    # estimate PACF via statsmodels' pacf if available, fallback to autocorr
    try:
        from statsmodels.tsa.stattools import pacf
        pacf_vals = list(pacf(train.dropna(), nlags=max_lag, method='ywunbiased')[1:])
    except Exception:
        pacf_vals = [train.dropna().autocorr(lag=lag) for lag in range(1, max_lag+1)]

    # choose p as first pacf lag exceeding threshold
    p = 0
    for i, v in enumerate(pacf_vals):
        if abs(v) > thresh:
            p = i + 1
            break
    # choose q as first acf lag exceeding threshold
    q = 0
    for i, v in enumerate(acf_vals):
        if abs(v) > thresh:
            q = i + 1
            break

    # Seasonal detection (weekly)
    weekly_ac = train.dropna().autocorr(lag=7)
    seasonal = False
    P = Q = D = 0
    seasonal_period = 7
    if weekly_ac is not None and abs(weekly_ac) > thresh:
        seasonal = True
        D = 1  # assume seasonal difference if weekly autocorrelation significant
        # set small seasonal P and Q based on acf/pacf at seasonal lags (7,14)
        try:
            pacf_seasonal = pacf(train.dropna(), nlags=seasonal_period*2, method='ywunbiased')
            # check lag 7 (index 7)
            if len(pacf_seasonal) > seasonal_period and abs(pacf_seasonal[seasonal_period]) > thresh:
                P = 1
            else:
                P = 0
        except Exception:
            P = 1
        # seasonal Q from acf at lag 7
        acf_seasonal = train.dropna().autocorr(lag=7)
        if acf_seasonal is not None and abs(acf_seasonal) > thresh:
            Q = 1
        else:
            Q = 0

    st.write(f"- Suggested non-seasonal order (p,d,q): **({p}, {d}, {q})**")
    if seasonal:
        st.write(f"- Suggested seasonal order (P,D,Q,s): **({P}, {D}, {Q}, {seasonal_period})** (weekly seasonality detected)")
    else:
        st.write("- No clear weekly seasonality detected (seasonal order not suggested)")

    # Allow user to override suggested orders if desired (optional UI)
    with st.expander("🔧 Advanced: override suggested orders"):
        p_user = st.number_input("p (AR order)", value=int(p), min_value=0, max_value=10, step=1)
        d_user = st.number_input("d (differences)", value=int(d), min_value=0, max_value=2, step=1)
        q_user = st.number_input("q (MA order)", value=int(q), min_value=0, max_value=10, step=1)
        if seasonal:
            P_user = st.number_input("P (seasonal AR)", value=int(P), min_value=0, max_value=5, step=1)
            D_user = st.number_input("D (seasonal diff)", value=int(D), min_value=0, max_value=2, step=1)
            Q_user = st.number_input("Q (seasonal MA)", value=int(Q), min_value=0, max_value=5, step=1)
            s_user = st.number_input("s (seasonal period)", value=int(seasonal_period), min_value=2, max_value=365, step=1)
        else:
            P_user = Q_user = D_user = s_user = 0
        # assign final orders
        p_final, d_final, q_final = int(p_user), int(d_user), int(q_user)
        P_final, D_final, Q_final, s_final = int(P_user), int(D_user), int(Q_user), int(s_user)

    p_final = q_final = 2
    d_final = 1
    P_final = D_final = Q_final = 1
    s_final = 7
    # ----- FIT MODEL (based on selection) -----
    try:
        if model_choice == "ARIMA":
            st.write(f"Fitting ARIMA({p_final},{d_final},{q_final}) on log series...")
            model = ARIMA(train, order=(p_final, d_final, q_final))
            fitted = model.fit()
            forecast_log = fitted.forecast(steps=len(test))

        elif model_choice == "SARIMA":
            if seasonal:
                st.write(f"Fitting SARIMA({p_final},{d_final},{q_final}) x ({P_final},{D_final},{Q_final},{s_final}) on log series...")
                model = SARIMAX(train, order=(p_final, d_final, q_final),
                                seasonal_order=(P_final, D_final, Q_final, s_final))
            else:
                st.write(f"Fitting SARIMA({p_final},{d_final},{q_final}) (no seasonal terms detected)...")
                model = SARIMAX(train, order=(p_final, d_final, q_final))
            fitted = model.fit(disp=False)
            forecast_log = fitted.forecast(steps=len(test))

        elif model_choice == "Holt-Winters":
            st.write("Fitting Holt-Winters on raw scale (log not used)...")
            # Holt-Winters expects raw scale so use original views series
            ts_raw = page_data.set_index('date')['views'].asfreq('D').fillna(0)
            train_raw, test_raw = train_test_split_series(ts_raw, test_size=test_size)
            model = ExponentialSmoothing(train_raw, trend='add', seasonal='add', seasonal_periods=7)
            fitted = model.fit()
            forecast_log = None  # not applicable
            forecast_raw = fitted.forecast(len(test_raw))

        # ----- INVERSE TRANSFORM FORECAST -----
        if model_choice in ["ARIMA", "SARIMA"]:
            # forecast_log is in log scale -> invert
            forecast_orig = np.expm1(forecast_log)
            actual_orig = np.expm1(test)
            train_orig = np.expm1(train)
        else:
            forecast_orig = forecast_raw
            actual_orig = test_raw
            train_orig = train_raw

        # ----- PLOT ORIGINAL-SCALE FORECAST VS ACTUAL -----
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(train_orig.index, train_orig, label="Train (orig)", color='blue')
        ax2.plot(actual_orig.index, actual_orig, label="Test (orig)", color='orange')
        ax2.plot(actual_orig.index, forecast_orig, label="Forecast (orig)", color='green')
        ax2.set_title(f"{model_choice} Forecast — {forecast_page} (original scale)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Views")
        ax2.legend()
        st.pyplot(fig2)

        # ----- METRICS (on original scale) -----
        if model_choice in ["ARIMA", "SARIMA"]:
            rmse_o, mae_o, mape_o = evaluate_forecast(actual_orig, forecast_orig)
        else:
            rmse_o, mae_o, mape_o = evaluate_forecast(actual_orig, forecast_orig)

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse_o:.2f}")
        col2.metric("MAE", f"{mae_o:.2f}")
        col3.metric("MAPE (%)", f"{mape_o:.2f}")

    except Exception as e:
        st.error(f"❌ Model failed during fit/forecast: {e}")


# ============================================
# 6️⃣ Optional Data View
# ============================================
with st.expander("📋 Show Cleaned Data for Selected Pages"):
    st.dataframe(df_long[df_long['Page'].isin(selected_pages)])
