# 📈 Bitcoin Price Forecasting using ARIMA & LSTM

A complete end-to-end **time series forecasting project** that compares a classical statistical model (ARIMA) with a deep learning model (LSTM) to predict Bitcoin prices.  
The project includes data preprocessing, exploratory data analysis, model training, evaluation, forecasting, and deployment using **Streamlit**.

---

## 🔍 Project Overview

- Forecast Bitcoin closing prices using historical data
- Compare **ARIMA** and **LSTM** model performance
- Generate a **30-day future price forecast**
- Deploy an interactive web application using Streamlit

---

## 🧠 Models Used

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- Suitable for stationary time-series data
- Captures short-term patterns and trends

### 2. LSTM (Long Short-Term Memory)
- Deep learning model for sequential data
- Captures long-term dependencies in price movements

---

## 🛠 Tech Stack

- **Language:** Python  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Statistical Modeling:** Statsmodels  
- **Deep Learning:** TensorFlow / Keras  
- **Evaluation:** Scikit-learn  
- **Deployment:** Streamlit  
- **Version Control:** Git & GitHub  

---

## 📊 Workflow

1. Data collection using `yfinance`
2. Exploratory Data Analysis (EDA)
3. Stationarity check using ADF test
4. Differencing for stationarity
5. Outlier detection (Boxplots)
6. ARIMA model training & forecasting
7. LSTM model training & forecasting
8. Model evaluation (RMSE, MAE)
9. Model comparison
10. Deployment using Streamlit

---

## 📁 Project Structure
Crypto_Price_Forecast/
│
├── models/
│   ├── arima_model.pkl        # Saved ARIMA model
│   └── lstm_model.h5          # Trained LSTM model
│
├── app.py                     # Streamlit application for forecasting & visualization
├── main.ipynb                 # Data analysis, preprocessing, modeling & evaluation
├── processed_crypto_data.csv  # Cleaned & preprocessed dataset
├── model_comparison.csv       # Model performance metrics (RMSE, MAE)
├── requirements.txt           # Project dependencies
├── .gitignore                 # Files & folders ignored by Git
└── README.md                  # Project documentation

---

## 📈 Results & Insights:

-ARIMA performs well for short-term forecasting
-LSTM captures long-term trends more effectively
-LSTM generally achieves lower error metrics compared to ARIMA

---

## 📌 Future Improvements:

-Hyperparameter tuning
-Multivariate forecasting
-Add confidence intervals
-Use real-time price updates

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
