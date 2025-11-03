import pandas as pd
import numpy as np
import os

# === 1. Veri yükleme ===
path = "/content/drive/MyDrive/anomaly_project/data/yfinance_clean.csv"
df = pd.read_csv(path)

print("Orijinal veri boyutu:", df.shape)
print(df.head())

# === 2. Daha düşük eşik (anomali oranını artır) ===
threshold = np.percentile(df["volatility"], 90)
df["label"] = (df["volatility"] > threshold).astype(int)

print(f"Yeni anomaly threshold (90th percentile): {threshold:.6f}")
print("Label dağılımı:", df["label"].value_counts().to_dict())

# === 3. Genişletilmiş feature listesi ===
features = ["log_return", "volatility", "Close", "High", "Low", "Open"]
X = df[features].values
y = df["label"].values

print("Yeni feature set boyutu:", X.shape)

# === 4. Windowing (128 uzunluk) ===
window_size = 128
X_windowed, y_windowed = [], []

for i in range(len(X) - window_size):
    X_windowed.append(X[i:i+window_size])
    y_windowed.append(y[i+window_size])

X_windowed = np.array(X_windowed)
y_windowed = np.array(y_windowed)

print("✅ Pencereleme tamamlandı!")
print("Yeni X shape:", X_windowed.shape)
print("Yeni y shape:", y_windowed.shape)
print("Label oranı:", {int(k): int(v) for k,v in zip(*np.unique(y_windowed, return_counts=True))})

# === 5. Kaydet ===
base_dir = "/content/drive/MyDrive/anomaly_project/data/windowed"
os.makedirs(base_dir, exist_ok=True)

np.save(os.path.join(base_dir, "X_finance_windowed_v2.npy"), X_windowed)
np.save(os.path.join(base_dir, "y_finance_windowed_v2.npy"), y_windowed)

print("\n💾 Kaydedildi:")
print(os.path.join(base_dir, "X_finance_windowed_v2.npy"))
print(os.path.join(base_dir, "y_finance_windowed_v2.npy"))
