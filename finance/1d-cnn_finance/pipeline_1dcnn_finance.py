import numpy as np
from sklearn.preprocessing import StandardScaler

# v2 verisini yükle
X_v2 = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/X_finance_windowed_v2.npy")
y_v2 = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/y_finance_windowed_v2.npy")

# Mevcut feature’ları normalize et
scaler = StandardScaler()
X_scaled = np.empty_like(X_v2)
for i in range(X_v2.shape[1]):
    X_scaled[:, i, :] = scaler.fit_transform(X_v2[:, i, :])

# Yeni feature’lar ekle: volatility change, rolling diff, jump ratio
vol_change = X_scaled[:, :, 1] - np.roll(X_scaled[:, :, 1], 1, axis=1)
roll_diff = X_scaled[:, :, 2] - X_scaled[:, :, 2].mean(axis=1, keepdims=True)
jump_ratio = (X_scaled[:, :, 3] - X_scaled[:, :, 4]) / (X_scaled[:, :, 5] + 1e-6)

# Üçünü birleştir
X_v3 = np.concatenate([X_scaled, 
                       vol_change[..., np.newaxis], 
                       roll_diff[..., np.newaxis], 
                       jump_ratio[..., np.newaxis]], axis=2)

print("✨ Enriched shape:", X_v3.shape)

# Kaydet
np.save("/content/drive/MyDrive/anomaly_project/data/windowed/X_finance_windowed_v3.npy", X_v3)
np.save("/content/drive/MyDrive/anomaly_project/data/windowed/y_finance_windowed_v3.npy", y_v2)

print("✅ Saved enriched v3 dataset to Drive.")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os

# === 1. VERİYİ YÜKLE ===
X_path = "/content/drive/MyDrive/anomaly_project/data/windowed/X_finance_windowed_v2.npy"
y_path = "/content/drive/MyDrive/anomaly_project/data/windowed/y_finance_windowed_v2.npy"

X = np.load(X_path)
y = np.load(y_path)

# Güvenlik: uzunluk eşitle
if len(y) > len(X):
    y = y[:len(X)]
elif len(y) < len(X):
    X = X[:len(y)]

print("Orijinal:", X.shape, y.shape)

# === 2. İLK NORMALİZASYON (eski 6 feature için) ===
scaler0 = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[2])
X_scaled = scaler0.fit_transform(X_reshaped).reshape(X.shape)
print("✅ İlk 6 feature normalize edildi:", X_scaled.shape)

# === 3. FEATURE ENGINEERING ===
X_new = []
for window in X_scaled:
    df = pd.DataFrame(window, columns=[f"f{i}" for i in range(window.shape[1])])

    # volatility change rate
    vol = df.std(axis=1)
    vol_rate = vol.diff().fillna(0)

    # rolling mean diff
    roll_mean = df.mean(axis=1).rolling(5, min_periods=1).mean()
    roll_diff = roll_mean.diff().fillna(0)

    # price jump ratio (son - ilk)
    jump_ratio = (df.iloc[:, -1] - df.iloc[:, 0]) / (np.abs(df.iloc[:, 0]) + 1e-6)

    # 3 yeni feature'ı birleştir
    new_features = np.vstack([vol_rate, roll_diff, jump_ratio]).T
    X_new.append(np.hstack([window, new_features]))

X_enriched = np.array(X_new)
print("✨ Enriched shape (6 + 3 feature):", X_enriched.shape)   # (N, 64, 9)

# === 4. İKİNCİ NORMALİZASYON (9 feature'ın tamamına) ===
scaler1 = StandardScaler()
X_enr_reshaped = X_enriched.reshape(-1, X_enriched.shape[2])
X_enriched_scaled = scaler1.fit_transform(X_enr_reshaped).reshape(X_enriched.shape)
print("✅ Tüm 9 feature yeniden normalize edildi:", X_enriched_scaled.shape)

# === 5. SADECE NORMAL ÖRNEKLERLE EĞİT ===
X_normal = X_enriched_scaled[y == 0]
X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Val:", X_val.shape)

# === 6. MODEL TANIMI (v6.1) ===
input_shape = (X_train.shape[1], X_train.shape[2])
inputs = Input(shape=input_shape)

# Encoder
x = Conv1D(128, 7, activation="relu", padding="same")(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(2, padding="same")(x)
x = Dropout(0.3)(x)

x = Conv1D(64, 5, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
encoded = MaxPooling1D(2, padding="same")(x)

# Decoder
x = Conv1D(64, 5, activation="relu", padding="same")(encoded)
x = BatchNormalization()(x)
x = UpSampling1D(2)(x)
x = Dropout(0.3)(x)

x = Conv1D(128, 7, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = UpSampling1D(2)(x)

# 🔧 çıkış kanal sayısı: feature sayısı = input_shape[1] DEĞİL, input_shape[2]
decoded = Conv1D(X_train.shape[2], 3, activation="sigmoid", padding="same")(x)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
autoencoder.summary()

# === 7. CALLBACKS ===
model_dir = "/content/drive/MyDrive/anomaly_project/models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "1dcnn_autoencoder_finance_v6_1.keras")

checkpoint = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1)
earlystop   = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
reduce_lr   = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                min_lr=1e-5, verbose=1)

# === 8. EĞİTİM ===
history = autoencoder.fit(
    X_train, X_train,
    epochs=60,
    batch_size=64,
    validation_data=(X_val, X_val),
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1
)

# === 9. LOSS GRAFİĞİ ===
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.title("1D-CNN Autoencoder v6.1 (Feature + Scaled)")
plt.show()

# === 10. DEĞERLENDİRME ===
autoencoder = tf.keras.models.load_model(model_path)
recons = autoencoder.predict(X_enriched_scaled)
mse_all = np.mean(np.square(X_enriched_scaled - recons), axis=(1, 2))

# ROC & threshold
fpr, tpr, thresholds = roc_curve(y, mse_all)
roc_auc = auc(fpr, tpr)
J = tpr - fpr
best_idx = np.argmax(J)
best_threshold = thresholds[best_idx]
y_pred_opt = (mse_all > best_threshold).astype(int)

cm = confusion_matrix(y, y_pred_opt)
tn, fp, fn, tp = cm.ravel()
precision, recall, f1, _ = precision_recall_fscore_support(
    y, y_pred_opt, average="binary", zero_division=0
)

print(f"\n🎯 Best threshold = {best_threshold:.4f}")
print(f"AUC = {roc_auc:.4f}")
print(f"Precision = {precision:.3f} | Recall = {recall:.3f} | F1 = {f1:.3f}")
print(f"\nConfusion Matrix:\n{cm}")

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
plt.scatter(fpr[best_idx], tpr[best_idx], color="red", label="Best Threshold", zorder=5)
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Autoencoder v6.1 (Feature-Enriched & Scaled)")
plt.legend()
plt.show()

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

# === Modeli ve veriyi yükle ===
autoencoder = load_model("/content/drive/MyDrive/anomaly_project/models/1dcnn_autoencoder_finance_v6_1.keras")
X = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/X_finance_windowed_v3.npy")
y = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/y_finance_windowed_v3.npy")


# === Reconstruction Error Hesapla ===
recons = autoencoder.predict(X)
mse_all = np.mean(np.square(X - recons), axis=(1, 2))

print("Reconstruction error shape:", mse_all.shape)
print("Example:", mse_all[:5])

# Flatten edilmiş özellik seti (her pencere ortalaması alınabilir)
X_flat = X.mean(axis=1)  # (5102, 9)

# Reconstruction error ekle
import pandas as pd
df_features = pd.DataFrame(X_flat, columns=[f"feat_{i}" for i in range(X_flat.shape[1])])
df_features["recon_error"] = mse_all
df_features["label"] = y

print(df_features.head())
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Veri böl
X_train, X_test, y_train, y_test = train_test_split(
    df_features.drop("label", axis=1), df_features["label"],
    test_size=0.2, random_state=42, stratify=df_features["label"]
)

# Model
xgb = XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42
)
xgb.fit(X_train, y_train)

# Tahminler
y_pred_prob = xgb.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

# Performans
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"\n🎯 Hybrid Model AUC: {roc_auc:.3f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'Hybrid ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Hybrid (Autoencoder + XGBoost)')
plt.legend()
plt.show()

import joblib
import os

# === Model klasörü ===
model_dir = "/content/drive/MyDrive/anomaly_project/models"
os.makedirs(model_dir, exist_ok=True)

# === XGBoost modelini kaydet ===
hybrid_model_path = os.path.join(model_dir, "Hybrid_XGBoost_v7.joblib")
joblib.dump(xgb, hybrid_model_path)

print(f"✅ Hybrid XGBoost v7 kaydedildi: {hybrid_model_path}")

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# === Modelleri yükle ===
auto_path = "/content/drive/MyDrive/anomaly_project/models/1dcnn_autoencoder_finance_v6_1.keras"
xgb_path = "/content/drive/MyDrive/anomaly_project/models/Hybrid_XGBoost_v7.joblib"

autoencoder = load_model(auto_path)
xgb_model = joblib.load(xgb_path)

print("✅ Autoencoder ve XGBoost yüklendi!")

# === Tek adımda anomali tespiti ===
def detect_anomaly(X_input):
    """
    X_input: shape = (num_samples, 64, 9)
    """
    # Reconstruction error
    recons = autoencoder.predict(X_input)
    mse = np.mean(np.square(X_input - recons), axis=(1, 2))

    # Flatten + merge
    X_flat = X_input.mean(axis=1)
    df_infer = pd.DataFrame(X_flat, columns=[f"feat_{i}" for i in range(X_flat.shape[1])])
    df_infer["recon_error"] = mse

    # XGBoost prediction
    y_prob = xgb_model.predict_proba(df_infer)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    return y_pred, y_prob, mse
# Test et (örnek olarak mevcut X_v3 datasından 10 pencere)
X_test_sample = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/X_finance_windowed_v3.npy")[:10]

y_pred, y_prob, mse = detect_anomaly(X_test_sample)

print("🔍 Tahminler:", y_pred)
print("📈 İhtimaller:", np.round(y_prob, 3))
print("⚙️ Reconstruction MSE:", np.round(mse, 3))

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support

def interactive_evaluate(X, y, default_threshold=0.5):
    """
    Interaktif eşik slider'ı ile model performansını görselleştirir.
    """
    y_pred_prob_full = detect_anomaly(X)[1]
    mse = detect_anomaly(X)[2]

    @interact(threshold=FloatSlider(value=default_threshold, min=0.1, max=0.9, step=0.05, description='Threshold'))
    def update(threshold):
        y_pred = (y_pred_prob_full > threshold).astype(int)

        # --- metrikler ---
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
        auc = roc_auc_score(y, y_pred_prob_full)
        cm = confusion_matrix(y, y_pred)

        print(f"\n🎯 Threshold = {threshold:.2f}")
        print(f"Precision = {precision:.3f} | Recall = {recall:.3f} | F1 = {f1:.3f} | AUC = {auc:.3f}")
        print("Confusion Matrix:")
        print(cm)

        # --- görselleştirme ---
        plt.figure(figsize=(12,5))
        plt.plot(mse, color='blue', alpha=0.6, label="Reconstruction Error")
        plt.scatter(np.where(y_pred==1), mse[y_pred==1], color='red', s=20, label="Predicted Anomaly")
        plt.scatter(np.where(y==1), mse[y==1], color='orange', marker='x', label="True Anomaly")
        plt.title(f"Anomaly Detection – Interactive Threshold = {threshold:.2f}")
        plt.xlabel("Window Index")
        plt.ylabel("Reconstruction Error")
        plt.legend()
        plt.show()
X = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/X_finance_windowed_v3.npy")
y = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/y_finance_windowed_v3.npy")

interactive_evaluate(X, y, default_threshold=0.5)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_final_visualization(X, y):
    """
    Hybrid Autoencoder + XGBoost modelinin anomaly score ve distribution grafikleri.
    """
    # --- Modelden tahminleri al ---
    y_pred, y_prob, mse = detect_anomaly(X)
    threshold = 0.5  # optimal threshold (Youden's J veya slider'dan alınabilir)
    
    # --- Anomaly intensity map (probability-based) ---
    plt.figure(figsize=(14, 7))
    
    # Üst panel: renkli heatmap benzeri anomaly distribution
    plt.subplot(2, 1, 1)
    sns.heatmap(y_prob[np.newaxis, :], cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)
    plt.title("🧠 Anomaly Intensity Over Time (Red = High Risk)")
    plt.ylabel("Anomaly Intensity")
    
    # Alt panel: reconstruction error + predicted vs true anomalies
    plt.subplot(2, 1, 2)
    plt.plot(y_prob, color='purple', lw=1.5, label="Anomaly Probability")
    plt.axhline(threshold, color='gray', linestyle='--', label=f"Threshold = {threshold:.2f}")
    plt.scatter(np.where(y==1), y_prob[y==1], color='orange', s=25, marker='x', label="True Anomaly")
    plt.scatter(np.where(y_pred==1), y_prob[y_pred==1], color='red', s=20, label="Predicted Anomaly")
    plt.xlabel("Window Index")
    plt.ylabel("Anomaly Score")
    plt.title("⚙️ Hybrid Model — Probability & Threshold View")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
X = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/X_finance_windowed_v3.npy")
y = np.load("/content/drive/MyDrive/anomaly_project/data/windowed/y_finance_windowed_v3.npy")

plot_final_visualization(X, y)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

def export_final_auto_report(X, y, threshold=0.5):
    # === Path & Date ===
    save_path = "/content/drive/MyDrive/anomaly_project/reports"
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, "anomaly_detection_report_final_auto.pdf")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # === Model predictions ===
    y_pred, y_prob, _ = detect_anomaly(X)

    # === Auto metrics ===
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # === Figure layout ===
    fig = plt.figure(figsize=(14, 11), facecolor='white')
    gs = fig.add_gridspec(4, 1, height_ratios=[0.3, 1.2, 1.2, 0.7])

    # === 0️⃣ Title ===
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    ax_title.text(
        0.5, 0.5,
        "💹 Unsupervised Financial Anomaly Detection using 1D-CNN Autoencoder + XGBoost",
        ha='center', va='center', fontsize=17, fontweight='bold', color='#1F3C88'
    )

    # === 1️⃣ Heatmap ===
    ax1 = fig.add_subplot(gs[1])
    sns.heatmap(y_prob[np.newaxis, :], cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False, ax=ax1)
    ax1.set_title("🔥 Anomaly Intensity Over Time (Red = High Risk)", fontsize=13, fontweight='bold')

    # === 2️⃣ Probability ===
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(y_prob, color='purple', lw=1.8, label="Anomaly Probability")
    ax2.axhline(threshold, color='gray', linestyle='--', lw=1.2, label=f"Threshold = {threshold:.2f}")
    ax2.scatter(np.where(y==1), y_prob[y==1], color='orange', s=30, marker='x', label="True Anomaly")
    ax2.scatter(np.where(y_pred==1), y_prob[y_pred==1], color='red', s=25, label="Predicted Anomaly")
    ax2.legend(loc='upper right')
    ax2.set_xlabel("Window Index")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_title("⚙️ Probability and Threshold Visualization", fontsize=13, fontweight='bold')

    # === 3️⃣ Metrics Table & Signature ===
    ax3 = fig.add_subplot(gs[3])
    ax3.axis('off')

    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-score", "AUC", "Threshold"],
        "Value": [f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", f"{auc:.3f}", f"{threshold:.2f}"]
    })

    table = ax3.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center',
        colColours=["#f4f4f4", "#f4f4f4"]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)

    ax3.text(0.02, -0.25, f"📅 Generated on: {date_str}", fontsize=10, color='gray', transform=ax3.transAxes)
    ax3.text(0.68, -0.25, "👩‍💻 Prepared by: Bengisu Göksu", fontsize=11, fontweight='bold', transform=ax3.transAxes)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    print(f"✅ Final automatic report Drive’a kaydedildi: {filename}")
    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
    print(f"Confusion Matrix:\n{cm}")

# === Çalıştır ===
export_final_auto_report(X, y, threshold=0.50)
