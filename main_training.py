import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

from lstm_model import train_lstm, predict_lstm

# -----------------------------
# LOAD DATA (CSV ONLY)
# -----------------------------
df = pd.read_csv("sales_data.csv")

print("✅ Data Loaded Successfully!")
print("Columns:", df.columns)

# -----------------------------
# RENAME COLUMNS (EDIT IF NEEDED)
# -----------------------------
df.columns = ['date', 'state', 'sales']

df['date'] = pd.to_datetime(df['date'])

models = {}

# -----------------------------
# PROCESS EACH STATE
# -----------------------------
for state in df['state'].unique():

    print(f"\n🔹 Processing State: {state}")

    data = df[df['state'] == state].copy()

    # SET DATE INDEX
    data = data.set_index('date').resample('D').sum()

    # HANDLE MISSING VALUES
    data['sales'] = data['sales'].ffill()

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    data['lag_1'] = data['sales'].shift(1)
    data['lag_7'] = data['sales'].shift(7)
    data['lag_30'] = data['sales'].shift(30)

    data['rolling_mean'] = data['sales'].rolling(7).mean()
    data['rolling_std'] = data['sales'].rolling(7).std()

    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month

    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    data.dropna(inplace=True)

    # -----------------------------
    # TRAIN TEST SPLIT
    # -----------------------------
    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    test = data.iloc[split:]

    X_train = train.drop('sales', axis=1)
    y_train = train['sales']

    X_test = test.drop('sales', axis=1)
    y_test = test['sales']

    errors = {}

    # -----------------------------
    # 1. XGBOOST
    # -----------------------------
    try:
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        errors['xgb'] = mean_absolute_error(y_test, pred_xgb)
    except Exception as e:
        print("❌ XGBoost Error:", e)
        errors['xgb'] = float('inf')

    # -----------------------------
    # 2. SARIMA
    # -----------------------------
    try:
        sarima = SARIMAX(train['sales'],
                         order=(1,1,1),
                         seasonal_order=(1,1,1,7))
        sarima_fit = sarima.fit(disp=False)
        pred_sarima = sarima_fit.forecast(len(test))
        errors['sarima'] = mean_absolute_error(y_test, pred_sarima)
    except Exception as e:
        print("❌ SARIMA Error:", e)
        errors['sarima'] = float('inf')

    # -----------------------------
    # 3. PROPHET
    # -----------------------------
    try:
        p_df = train.reset_index()
        p_df.columns = ['ds', 'y']

        prophet = Prophet()
        prophet.fit(p_df)

        future = prophet.make_future_dataframe(periods=len(test))
        forecast = prophet.predict(future)

        pred_prophet = forecast['yhat'][-len(test):].values
        errors['prophet'] = mean_absolute_error(y_test, pred_prophet)
    except Exception as e:
        print("❌ Prophet Error:", e)
        errors['prophet'] = float('inf')

    # -----------------------------
    # 4. LSTM
    # -----------------------------
    try:
        lstm_model = train_lstm(train['sales'])
        pred_lstm = predict_lstm(lstm_model, train['sales'], len(test))
        errors['lstm'] = mean_absolute_error(y_test, pred_lstm)
    except Exception as e:
        print("❌ LSTM Error:", e)
        errors['lstm'] = float('inf')

    # -----------------------------
    # SELECT BEST MODEL
    # -----------------------------
    best_model = min(errors, key=errors.get)

    print("📊 Errors:", errors)
    print("🏆 Best Model:", best_model)

    if best_model == 'xgb':
        models[state] = ('xgb', xgb)
    elif best_model == 'sarima':
        models[state] = ('sarima', sarima_fit)
    elif best_model == 'prophet':
        models[state] = ('prophet', prophet)
    else:
        models[state] = ('lstm', lstm_model)

# -----------------------------
# SAVE MODELS
# -----------------------------
pickle.dump(models, open("best_models.pkl", "wb"))

print("\n🎉 Training Completed & Models Saved!")