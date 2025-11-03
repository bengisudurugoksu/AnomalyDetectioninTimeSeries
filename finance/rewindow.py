
import numpy as np
import os

# === 1. Kaynak dosyaları yükle ===
X = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/X_finance.npy")
y = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/y_finance.npy")

print("Orijinal X:", X.shape, "y:", y.shape)

# === 2. Pencereleme parametreleri ===
window_size = 128
step = 1

X_windowed = []
y_windowed = []

for i in range(0, len(X) - window_size, step):
    X_windowed.append(X[i:i + window_size])     # 128 ardışık örnek
    y_windowed.append(y[i + window_size - 1])   # son örneğin label’ı

X_windowed = np.array(X_windowed)
y_windowed = np.array(y_windowed)

print("✅ Pencereleme tamamlandı!")
print("Yeni X shape:", X_windowed.shape)
print("Yeni y shape:", y_windowed.shape)

# === 3. Kaydet ===
save_dir = "/content/drive/MyDrive/anomaly_project/data/windowed"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "X_finance_windowed.npy"), X_windowed)
np.save(os.path.join(save_dir, "y_finance_windowed.npy"), y_windowed)

print("💾 Kaydedildi:")
print("X_finance_windowed.npy →", os.path.join(save_dir, "X_finance_windowed.npy"))
print("y_finance_windowed.npy →", os.path.join(save_dir, "y_finance_windowed.npy"))
