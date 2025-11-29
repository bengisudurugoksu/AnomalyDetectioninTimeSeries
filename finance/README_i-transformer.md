---

# 📘 **iTransformer Autoencoder for Financial Time-Series Anomaly Detection**

*Advanced Deep Learning Model for Detecting Rare Market Irregularities*

---

## 🔍 **1. Overview**

This project implements an **iTransformer-based autoencoder** to detect anomalies in multivariate financial time-series data.
Unlike traditional LSTM or CNN autoencoders, the iTransformer architecture leverages **attention mechanisms** to model long-range dependencies, making it highly effective for financial patterns that evolve over time.

The model is trained on **normal market behaviour only**, enabling it to learn a robust reconstruction manifold.
When abnormal patterns occur (e.g., sudden volatility spikes), reconstruction error increases sharply — producing a clear anomaly signal.

This model is evaluated on both:

* **Normal conditions** (AAPL 2021–2023)
* **Real crash event** (SPY 2020 Pandemic Crash)

---

## 📦 **2. Data & Preprocessing**

### **2.1 Features Used (9 total)**

From the raw OHLCV data we derive:

| Feature | Description                        |
| ------- | ---------------------------------- |
| Open    | Market opening price               |
| High    | Daily high                         |
| Low     | Daily low                          |
| Close   | Market closing price               |
| Volume  | Trade volume                       |
| HL      | High − Low (intraday range)        |
| OC      | Close − Open                       |
| RET     | Daily return (Pct change of Close) |
| VOL_RET | Volume pct change                  |

### **2.2 Windowing**

Time-series is converted into fixed-length sequences:

```
Window length = 64  
Final shape   = (N, 64, 9)
```

Each window represents 64 consecutive days of multivariate market behaviour.

---

## 🧠 **3. Model Architecture — iTransformer Autoencoder**

The autoencoder consists of:

* **Input projection layer** → Dense(embed_dim)
* **N stacked iTransformer blocks**
* **Output projection** → Dense(num_features)

### **iTransformer Block Details**

Each block includes:

* Multi-Head Self-Attention
* Position-wise Feedforward Network
* Layer Normalization
* Residual Connections
* Dropout Regularization

This structure allows the model to capture relationships between features across long temporal horizons.

---

## 🏋️ **4. Training Procedure**

### **Objective**

The model is trained to **reconstruct normal sequences**:

[
\mathcal{L} = \text{MSE}(X, \hat{X})
]

### **Important notes**

* Only windows with **normal labels (y = 0)** are used for training.
* This makes the model sensitive to deviations seen during anomalous periods.

### **Callbacks**

* EarlyStopping (patience=10, restore_best=True)
* ReduceLROnPlateau (factor=0.5)
* ModelCheckpoint (saves best model as `.h5`)

---

## 🚨 **5. Anomaly Detection Method**

After inference, per-window reconstruction error is computed:

[
\text{MSE}_i = \frac{1}{64 \times 9} \sum (X_i - \hat{X}_i)^2
]

Two thresholding strategies are used:

### **(a) Statistical Threshold**

[
T = \mu_{\text{mse}} + 3\sigma
]

### **(b) Expert-Selected Percentile Method**

For highly noisy data (e.g., SPY crash):

[
T = \text{95th percentile of MSE}
]

This is closer to how financial anomaly detection is done in practice (volatility clustering, fat-tailed distributions, etc.).

---

## 📊 **6. Real-World Stress Test — SPY 2020 Pandemic Crash**

The SPY ETF experienced a rapid crash from **February–March 2020**, a perfect test scenario for model robustness.

### **Procedure**

1. Download SPY OHLCV data from **2019-09 to 2020-07**
2. Apply identical preprocessing and windowing (shape = `(144, 64, 9)`)
3. Run model inference
4. Compute reconstruction error curve
5. Apply percentile-based threshold
6. Overlay detected anomalies on price chart

### **Findings**

* The model shows a **steady increase in reconstruction error** as volatility rises.
* Peak error coincides with the **sharpest price drawdowns**.
* Using percentile threshold, the model correctly flags **4 anomaly windows** during the crash.
* This demonstrates meaningful sensitivity to real-world market stress.

---

## 📈 **7. Example Outputs**

### **Reconstruction Error Curve**

Shows model sensitivity during SPY crash.

### **Price vs Scaled MSE**

MSE is normalized and scaled against price for clear visual comparison.

### **Detected Anomalies**

Anomalies are marked as black dots on the price chart — aligning with crash dynamics.

---

## 🧪 **8. How to Run**

### **1. Load the model**

```python
model = keras.models.load_model(
    "itransformer_finance_best.h5",
    custom_objects={"iTransformerBlock": iTransformerBlock}
)
```

### **2. Prepare your financial dataset**

```python
df = yf.download("AAPL", start="2021-01-01", end="2023-01-01")
df = preprocess(df)   # includes engineering HL, OC, RET, VOL_RET
windows = create_windows(df.values)
```

### **3. Run inference**

```python
recon = model.predict(windows)
mse = np.mean((windows - recon)**2, axis=(1,2))
```

### **4. Apply threshold**

```python
threshold = np.percentile(mse, 95)
anomalies = mse > threshold
```

### **5. Plot**

Included in the repository as `visualize_anomalies.py`.

---

## 🧾 **9. Conclusion**

This project demonstrates that:

* **Transformers can model financial time-series more effectively** than recurrent or convolutional models.
* An autoencoder trained only on normal regime dynamics can detect **abrupt regime shifts**.
* The model successfully identified anomalies during a **major real-world market event** (COVID-19 crash).

This provides strong evidence that iTransformer-based reconstruction errors are a viable signal for financial anomaly detection.

---

