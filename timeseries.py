import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------------
# CONFIG
# -----------------------------------------
st.set_page_config(page_title="Forecast Dashboard", layout="wide")

st.title("📊 Multi-Series Forecasting Dashboard")
st.write("Interactive tool for demand forecasting and model comparison")

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data
def load_data():
    # Replace with your real dataset
    dates = pd.date_range(start="2015-01-01", periods=120, freq="MS")

    df = pd.DataFrame({
        "date": dates,
        "A1_TravelTumbler": 200 + 10*np.sin(2*np.pi*dates.month/12) + np.random.normal(0,10,120),
        "A2_ColdBrew": 150 + 15*np.sin(2*np.pi*dates.month/12) + np.random.normal(0,12,120),
        "A3_CoffeeBeans": 300 + 8*np.sin(2*np.pi*dates.month/12) + np.random.normal(0,15,120),
    })

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").asfreq("MS")
    df["Total"] = df.sum(axis=1)

    return df

df = load_data()

# -----------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------
st.sidebar.header("Controls")

series_name = st.sidebar.selectbox(
    "Select Series",
    df.columns
)

horizon = st.sidebar.slider(
    "Forecast Horizon",
    min_value=3,
    max_value=24,
    value=12
)

# Train/test split
train = df.iloc[:-horizon]
test = df.iloc[-horizon:]

# -----------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------
def create_features(series):
    df = pd.DataFrame({'y': series})

    df['lag_1'] = series.shift(1)
    df['lag_12'] = series.shift(12)

    df['month'] = series.index.month
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)

    df['trend'] = np.arange(len(df))
    df['roll_mean_12'] = series.shift(1).rolling(12).mean()

    return df.dropna()

# -----------------------------------------
# MODELS
# -----------------------------------------
def hw_forecast(series, horizon):
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=12
    ).fit()

    forecast = model.forecast(horizon)
    return forecast, model


def arima_forecast(series, horizon):
    model = ARIMA(series, order=(1,1,1)).fit()
    forecast = model.get_forecast(steps=horizon)

    return forecast.predicted_mean, model


def ml_forecast(series, horizon, model_type="rf"):
    df_feat = create_features(series)

    X = df_feat.drop(columns=['y'])
    y = df_feat['y']

    model = RandomForestRegressor(n_estimators=300, random_state=42) \
        if model_type == "rf" else MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500)

    model.fit(X, y)

    history = series.copy()
    preds = []

    for i in range(horizon):
        temp = create_features(history)
        last_row = temp.iloc[-1:].drop(columns=['y'])

        pred = model.predict(last_row)[0]
        preds.append(pred)

        next_index = history.index[-1] + pd.offsets.MonthBegin(1)
        history.loc[next_index] = pred

    return pd.Series(preds, index=test.index)


# -----------------------------------------
# EVALUATION
# -----------------------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
    return mae, rmse, wmape


# -----------------------------------------
# RUN MODELS
# -----------------------------------------
y_train = train[series_name]
y_test = test[series_name]

hw_pred, hw_model = hw_forecast(y_train, horizon)
arima_pred, arima_model = arima_forecast(y_train, horizon)
rf_pred = ml_forecast(y_train, horizon, "rf")
mlp_pred = ml_forecast(y_train, horizon, "mlp")

# -----------------------------------------
# PLOT FORECASTS
# -----------------------------------------
st.subheader("📈 Forecast Visualization")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(train.index, y_train, label="Train")
ax.plot(test.index, y_test, label="Actual")

ax.plot(hw_pred.index, hw_pred, label="Holt-Winters")
ax.plot(arima_pred.index, arima_pred, label="ARIMA")
ax.plot(rf_pred.index, rf_pred, label="Random Forest")
ax.plot(mlp_pred.index, mlp_pred, label="Neural Network")

ax.legend()
ax.set_title(f"Forecast Comparison - {series_name}")

st.pyplot(fig)

# -----------------------------------------
# MODEL COMPARISON
# -----------------------------------------
st.subheader("📊 Model Performance")

results = []

for name, pred in {
    "Holt-Winters": hw_pred,
    "ARIMA": arima_pred,
    "Random Forest": rf_pred,
    "Neural Network": mlp_pred
}.items():
    mae, rmse, wmape = evaluate(y_test, pred)

    results.append({
        "Model": name,
        "MAE": round(mae,2),
        "RMSE": round(rmse,2),
        "wMAPE": round(wmape*100,2)
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df)

# -----------------------------------------
# RESIDUAL DIAGNOSTICS
# -----------------------------------------
st.subheader("🔬 Residual Diagnostics (ARIMA)")

residuals = arima_model.resid

fig2, ax = plt.subplots(1,2, figsize=(12,4))

sm.graphics.tsa.plot_acf(residuals, ax=ax[0])
sm.qqplot(residuals, line='s', ax=ax[1])

st.pyplot(fig2)

# -----------------------------------------
# SYSTEM VIEW
# -----------------------------------------
st.subheader("🔗 System-Level Forecast (Total)")

bottom_up = (
    hw_forecast(train["A1_TravelTumbler"], horizon)[0] +
    hw_forecast(train["A2_ColdBrew"], horizon)[0] +
    hw_forecast(train["A3_CoffeeBeans"], horizon)[0]
)

agg_hw, _ = hw_forecast(train["Total"], horizon)

fig3, ax = plt.subplots(figsize=(10,5))

ax.plot(test.index, test["Total"], label="Actual")
ax.plot(bottom_up.index, bottom_up, label="Bottom-Up")
ax.plot(agg_hw.index, agg_hw, label="Direct Aggregate")

ax.legend()
ax.set_title("System Forecast Comparison")

st.pyplot(fig3)
