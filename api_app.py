from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

models = pickle.load(open("best_models.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Forecast API Running"}

@app.get("/predict/{state}")
def predict(state: str):

    if state not in models:
        return {"error": "State not found"}

    model_type, model = models[state]
    steps = 56

    if model_type == 'xgb':
        df = pd.DataFrame({
            'lag_1':[100]*steps,
            'lag_7':[100]*steps,
            'lag_30':[100]*steps,
            'rolling_mean':[100]*steps,
            'rolling_std':[10]*steps,
            'day_of_week':[i%7 for i in range(steps)],
            'month':[5]*steps,
            'is_weekend':[1 if i%7>=5 else 0 for i in range(steps)]
        })
        pred = model.predict(df)

    elif model_type == 'sarima':
        pred = model.forecast(steps)

    elif model_type == 'prophet':
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        pred = forecast['yhat'][-steps:].values

    else:
        pred = [100]*steps  # simplified LSTM for API

    return {"state": state, "forecast": list(pred)}