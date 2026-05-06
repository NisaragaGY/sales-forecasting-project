import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_dataset(data, step=10):
    X, y = [], []
    for i in range(len(data)-step-1):
        X.append(data[i:(i+step)])
        y.append(data[i+step])
    return np.array(X), np.array(y)

def train_lstm(series):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.values.reshape(-1,1))

    X, y = create_dataset(data)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    model.fit(X, y, epochs=5, verbose=0)

    model.scaler = scaler
    return model

def predict_lstm(model, series, steps):
    scaler = model.scaler
    data = scaler.transform(series.values.reshape(-1,1))

    seq = data[-10:]
    preds = []

    for _ in range(steps):
        x = seq.reshape(1,10,1)
        pred = model.predict(x, verbose=0)
        preds.append(pred[0][0])
        seq = np.append(seq[1:], pred)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    return preds.flatten()