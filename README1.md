# Sales Forecasting System

## Overview
This project predicts the next 8 weeks of sales for each state using time series forecasting.

## Features
- Handles missing data
- Feature engineering (lag, rolling mean, time features)
- Multiple models: XGBoost, SARIMA, Prophet, LSTM
- Automatic best model selection
- REST API using FastAPI

## How to Run
1. Install requirements
2. Run training:
   python main_training.py
3. Run API:
   uvicorn api_app:app --reload

## Output
API returns forecasted sales values in JSON format.

## Author
Nisarga GY
