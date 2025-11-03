import pandas as pd
import numpy as np
import os

# === 1. Veriyi yükle ===
path = "/content/drive/MyDrive/anomaly_project/data/yfinance_clean.csv"
df = pd.read_csv(path)

print("Orijinal veri boyutu:", df.shape)
print(df.head())

# === 2. Volatility sütununa göre anomaly threshold belirle ===
threshold = np.percentile(df["volatility"], 95)   # En yüksek %5'i anomaly kabul et
print(f"Anomali eşiği (95. percentile): {threshold:.6f}")

# === 3. Label sütunu oluştur ===
df["label"] = (df["volatility"] > threshold).astype(int)

# === 4. Dağılımı kontrol et ===
print("Label dağılımı:", df["label"].value_counts().to_dict())

# === 5. Yeni X ve y oluştur (windowed değil, genel hazırlık için) ===
features = ["log_return", "volatility", "Close"]
X = df[features].values
y = df["label"].values

# === 6. Kaydet ===
base_dir = "/content/drive/MyDrive/anomaly_project/data/windowed"
os.makedirs(base_dir, exist_ok=True)

np.save(os.path.join(base_dir, "X_finance.npy"), X)
np.save(os.path.join(base_dir, "y_finance.npy"), y)

print("✅ Kaydedildi:")
print(os.path.join(base_dir, "X_finance.npy"))
print(os.path.join(base_dir, "y_finance.npy"))
