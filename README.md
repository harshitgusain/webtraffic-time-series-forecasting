# webtraffic-time-series-forecasting
This project focuses on time series forecasting using Wikipedia page view data. The objective is to analyze historical time-based data and predict future values using classical forecasting models. The project compares multiple models based on standard error and accuracy metrics.

The dataset used in this project is sourced from a Kaggle competition, which contains real-world Wikipedia traffic data.

📊 Dataset

Source: Kaggle (Wikipedia Web Traffic / Time Series Competition)

Type: Time-series data

Description: Historical page view counts of Wikipedia articles recorded over time.

⚙️ Models Implemented

The following forecasting models are used in this project:

ARIMA (AutoRegressive Integrated Moving Average)

SARIMAX (Seasonal ARIMA with Exogenous Variables)

Holt-Winters Exponential Smoothing

Each model is trained on historical data and used to forecast future values.

📈 Evaluation Metrics

To compare the performance of different models, the following evaluation metrics are calculated:

RMSE (Root Mean Squared Error)

MSE (Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

These metrics help in measuring prediction accuracy and model reliability.

🧪 Methodology

Data loading and preprocessing

Time series analysis and visualization

Model training (ARIMA, SARIMAX, Holt-Winters)

Forecast generation

Model evaluation using error metrics

Performance comparison across models

🛠️ Technologies Used

Python

NumPy

Pandas

Matplotlib / Seaborn

Statsmodels

Scikit-learn

📌 Results

The results show how different classical time series models perform on Wikipedia traffic data. Performance is evaluated based on error metrics, allowing comparison between seasonal and non-seasonal models.

🚀 Future Improvements

Implement deep learning models (LSTM, GRU)

Hyperparameter tuning for better accuracy

Cross-validation for time series

Multivariate forecasting
