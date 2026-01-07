import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Bitcoin Price Forecast", layout="wide")

st.title("📈 Bitcoin Price Forecasting App")
st.write("ARIMA vs LSTM comparison with 30-day forecast")

# -----------------------------
# LOAD DATA
# -----------------------------
st.subheader("Loading dataset...")

try:
    df = pd.read_csv("processed_crypto_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Close']]
    st.success("✅ Dataset loaded successfully")
except Exception as e:
    st.error("❌ Failed to load dataset")
    st.exception(e)
    st.stop()

# -----------------------------
# LOAD MODELS
# -----------------------------
st.subheader("Loading models...")

try:
    arima_model = joblib.load("models/arima_model.pkl")
    st.success("✅ ARIMA model loaded")
except Exception as e:
    st.error("❌ Failed to load ARIMA model")
    st.exception(e)
    st.stop()

try:
    lstm_model = load_model("models/lstm_model.h5")
    st.success("✅ LSTM model loaded")
except Exception as e:
    st.error("❌ Failed to load LSTM model")
    st.exception(e)
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Settings")
forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

EVAL_DAYS = 30

actual = df['Close'][-EVAL_DAYS:]
arima_eval = arima_model.forecast(steps=EVAL_DAYS)

rmse_arima = np.sqrt(mean_squared_error(actual, arima_eval, squared=False))
mae_arima = mean_absolute_error(actual, arima_eval)

metrics_df = pd.DataFrame({
    "Model": ["ARIMA", "LSTM"],
    "RMSE": [rmse_arima, 0],
    "MAE": [mae_arima, 0]
})
st.sidebar.subheader("Model Evaluation (Last 30 Days)")
st.sidebar.dataframe(metrics_df)

# -----------------------------
# FORECASTS
# -----------------------------
arima_forecast = arima_model.forecast(steps=forecast_days)

future_dates = pd.date_range(
    start=df.index[-1],
    periods=forecast_days + 1,
    freq="D"
)[1:]

# -----------------------------
# LSTM FORECAST
# -----------------------------
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']])

WINDOW_SIZE = 60
last_sequence = scaled_close[-WINDOW_SIZE:]
last_sequence = last_sequence.reshape(1, WINDOW_SIZE, 1)

lstm_predictions = []
current_sequence = last_sequence.copy()

for _ in range(forecast_days):
    next_pred = lstm_model.predict(current_sequence, verbose=0)
    lstm_predictions.append(next_pred[0, 0])

    current_sequence = np.append(
        current_sequence[:, 1:, :],
        next_pred.reshape(1, 1, 1),
        axis=1
    )

lstm_predictions = np.array(lstm_predictions).reshape(-1, 1)
lstm_forecast = scaler.inverse_transform(lstm_predictions).flatten()

# -----------------------------
# TABS FOR PLOTS (KEY FIX)
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 ARIMA Forecast",
    "🤖 LSTM Forecast",
    "📊 Comparison"
])

# -------- TAB 1: ARIMA --------
with tab1:
    st.subheader("ARIMA Forecast vs Actual")

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df.index, df['Close'], label="Actual", color="black")
    ax1.plot(future_dates, arima_forecast, label="ARIMA Forecast", color="blue")
    ax1.legend()
    ax1.set_title("ARIMA Forecast")

    st.pyplot(fig1)

# -------- TAB 2: LSTM --------
with tab2:
    st.subheader("LSTM Forecast vs Actual")

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df.index, df['Close'], label="Actual", color="black")
    ax2.plot(future_dates, lstm_forecast, label="LSTM Forecast", color="red")
    ax2.legend()
    ax2.set_title("LSTM Forecast")

    st.pyplot(fig2)

# -------- TAB 3: COMPARISON --------
with tab3:
    st.subheader("ARIMA vs LSTM Comparison")

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(df.index, df['Close'], label="Actual", color="black")
    ax3.plot(future_dates, arima_forecast, label="ARIMA", color="blue")
    ax3.plot(future_dates, lstm_forecast, label="LSTM", color="red")
    ax3.legend()
    ax3.set_title("Model Comparison")

    st.pyplot(fig3)

# -----------------------------
# COMPARISON TABLE
# -----------------------------
st.subheader("📊 Model Comparison")

try:
    comparison_df = pd.read_csv("model_comparison.csv")
    st.dataframe(comparison_df)
except Exception as e:
    st.error("❌ Failed to load comparison table")
    st.exception(e)
    
# -----------------------------
# Save Forecasts
# -----------------------------
Forecast_df = pd.DataFrame({
    "Date": future_dates,
    "ARIMA_Forecast": arima_forecast.values,
    "LSTM_Forecast": lstm_forecast
})

Forecast_df.to_csv("crypto_forecasts.csv", index=False)

st.success("✅ Forecasts saved to crypto_forecasts.csv")
st.dataframe(Forecast_df)