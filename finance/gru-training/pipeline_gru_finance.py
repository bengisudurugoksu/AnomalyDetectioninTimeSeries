import numpy as np
import pandas as pd
import yfinance as yf

# Load ETH data
df = yf.download(
    "ETH-USD",
    start="2017-01-01",
    end="2021-12-31",
    interval="1d",
    auto_adjust=False,
    progress=False
).reset_index()


df = df[["Date", "Close"]].dropna().reset_index(drop=True)

# log-return
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna().reset_index(drop=True)

print(df.head())
import os, json, joblib

ARTIFACT_DIR = "/content/drive/MyDrive/anomaly_project/artifacts_gru"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
# chronological split
split_ratio = 0.7
split_idx = int(len(df) * split_ratio)

train_returns = df["log_return"].values[:split_idx]
test_returns  = df["log_return"].values[split_idx:]

train_dates = df["Date"].values[:split_idx]
test_dates  = df["Date"].values[split_idx:]
print("Train returns:", train_returns.shape)
print("Test returns :", test_returns.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_returns.reshape(-1, 1)).ravel()
test_scaled  = scaler.transform(test_returns.reshape(-1, 1)).ravel()

print("Scaled train mean/std:", train_scaled.mean(), train_scaled.std())
joblib.dump(
    scaler,
    f"{ARTIFACT_DIR}/gru_return_scaler.joblib"
)
WINDOW = 32
meta = {
    "model": "GRU",
    "window": WINDOW,
    "split_ratio": split_ratio,
    "task": "log-return next-step forecasting",
    "anomaly_method": "residual percentile",
    "threshold_percentile": 97
}

with open(
    f"{ARTIFACT_DIR}/gru_return_meta.json",
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

# GRU input shape
X_train = X_train[..., None]
X_test  = X_test[..., None]

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)

inputs = Input(shape=(WINDOW, 1))
x = GRU(64)(inputs)
out = Dense(1)(x)

model = Model(inputs, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=1
)
model.save(
    f"{ARTIFACT_DIR}/gru_return_anomaly_w32.keras"
)

# predictions
yhat_tr = model.predict(X_train, verbose=0).ravel()
yhat_te = model.predict(X_test,  verbose=0).ravel()

# residuals
res_tr = np.abs(y_train - yhat_tr)
res_te = np.abs(y_test  - yhat_te)

print("Train residual mean/std:", res_tr.mean(), res_tr.std())
print("Test  residual mean/std:", res_te.mean(), res_te.std())
thr = np.percentile(res_tr, 97)   # sonra 97 ve 98 de deneyeceğiz
with open(
    f"{ARTIFACT_DIR}/gru_threshold_p97.txt",
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

plt.title("ETH Log-return Anomaly Detection (GRU)")
plt.xlabel("Date")
plt.ylabel("Log-return")
plt.legend()
plt.tight_layout()
plt.show()
# test döneminin price'ı
test_close = df["Close"].values[split_idx:]

# window yüzünden ilk WINDOW günü düşür
test_close_aligned = test_close[WINDOW:]

# dates zaten vardı
dates_test = test_dates[WINDOW:]

print(len(test_close_aligned), len(anomaly))
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15,5))

# price
plt.plot(dates_test, test_close_aligned,
         label="ETH Close Price (test)",
         linewidth=2)

# anomaly noktaları
idx = np.where(anomaly == 1)[0]
plt.scatter(
    dates_test[idx],
    test_close_aligned[idx],
    color="red",
    s=35,
    label="Anomaly (return-based)"
)

plt.title("ETH Price vs Anomaly (GRU return-based)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
for q in [95, 97, 99]:
    thr_q = np.percentile(res_tr, q)
    anomaly_q = (res_te > thr_q)
    overlap = np.sum(big & anomaly_q)
    print(f"thr p{q}: anomalies={anomaly_q.mean():.3f}, big detected={overlap}/{big.sum()}")
# naive prediction
yhat_naive = X_test[:, -1, 0]   # son return

res_naive = np.abs(y_test - yhat_naive)
res_gru   = np.abs(y_test - yhat_te)

print("Naive MAE:", res_naive.mean())
print("GRU   MAE:", res_gru.mean())
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.hist(res_gru, bins=50, alpha=0.7, label="GRU residual")
plt.hist(res_naive, bins=50, alpha=0.5, label="Naive residual")
plt.legend()
plt.title("Residual Distribution Comparison")
plt.show()
