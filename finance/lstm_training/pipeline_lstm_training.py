import numpy as np
import pandas as pd
import yfinance as yf

df = yf.download(
    "ETH-USD",   # TSLA-USD yapabilirsin
    start="2017-01-01",
    end="2021-12-31",
    interval="1d",
    auto_adjust=False,
    progress=False
).reset_index()

df = df[["Date", "Close"]].dropna().reset_index(drop=True)

df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna().reset_index(drop=True)
split_ratio = 0.7
split_idx = int(len(df) * split_ratio)

train_returns = df["log_return"].values[:split_idx]
test_returns  = df["log_return"].values[split_idx:]

train_dates = df["Date"].values[:split_idx]
test_dates  = df["Date"].values[split_idx:]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_returns.reshape(-1,1)).ravel()
test_scaled  = scaler.transform(test_returns.reshape(-1,1)).ravel()
import joblib, os

ARTIFACT_DIR = "/content/drive/MyDrive/anomaly_project/artifacts_lstm"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

joblib.dump(
    scaler,
    f"{ARTIFACT_DIR}/lstm_return_scaler.joblib"
)
WINDOW = 32

import json

meta = {
    "window": WINDOW,
    "split_ratio": split_ratio,
    "task": "LSTM log-return next-step forecasting",
    "anomaly_method": "residual + percentile threshold"
}

with open(
    f"{ARTIFACT_DIR}/lstm_return_meta.json",
    "w"
) as f:
    json.dump(meta, f, indent=2)


def make_windows(x, w):
    X, y = [], []
    for i in range(len(x) - w):
        X.append(x[i:i+w])
        y.append(x[i+w])
    return np.array(X), np.array(y)

X_train, y_train = make_windows(train_scaled, WINDOW)
X_test,  y_test  = make_windows(test_scaled,  WINDOW)

X_train = X_train[..., None]
X_test  = X_test[..., None]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)

inputs = Input(shape=(WINDOW, 1))
x = LSTM(64)(inputs)      # ⬅️ TEK FARK BU
out = Dense(1)(x)

model = Model(inputs, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

es = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=1
)
model.save(
    f"{ARTIFACT_DIR}/lstm_return_anomaly_w32.keras"
)

yhat_tr = model.predict(X_train, verbose=0).ravel()
yhat_te = model.predict(X_test,  verbose=0).ravel()

res_tr = np.abs(y_train - yhat_tr)
res_te = np.abs(y_test  - yhat_te)

print("Train residual mean/std:", res_tr.mean(), res_tr.std())
print("Test  residual mean/std:", res_te.mean(), res_te.std())
thr = np.percentile(res_tr, 97)
with open(
    f"{ARTIFACT_DIR}/lstm_threshold_p97.txt",
    "w"
) as f:
    f.write(str(thr))

anomaly = (res_te > thr).astype(int)

print("Threshold:", thr)
print("Anomaly rate:", anomaly.mean())
dates_test = test_dates[WINDOW:]

import matplotlib.pyplot as plt

plt.figure(figsize=(14,4))
plt.plot(dates_test, y_test, label="Log-return", alpha=0.7)

idx = np.where(anomaly == 1)[0]
plt.scatter(dates_test[idx], y_test[idx],
            color="red", s=25, label="Anomaly")

plt.title("Log-return Anomaly Detection (LSTM)")
plt.xlabel("Date")
plt.ylabel("Log-return")
plt.legend()
plt.tight_layout()
plt.show()
price_test = df["Close"].values[split_idx+WINDOW:]

plt.figure(figsize=(14,4))
plt.plot(dates_test, price_test, label="Price")

plt.scatter(dates_test[idx], price_test[idx],
            color="red", s=25, label="Anomaly")

plt.title("Price vs Anomaly (LSTM, return-based)")
plt.legend()
plt.tight_layout()
plt.show()
# NAIVE: y[t] ≈ y[t-1]
yhat_naive = X_test[:, -1, 0]   # window'ın son return'ü
from sklearn.metrics import mean_absolute_error

mae_naive = mean_absolute_error(y_test, yhat_naive)
mae_lstm  = mean_absolute_error(y_test, yhat_te)

print("NAIVE MAE:", mae_naive)
print("LSTM  MAE:", mae_lstm)
