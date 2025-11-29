# **iTransformer Autoencoder for Multivariate Financial Time-Series Anomaly Detection**

**Author:** *Bengisu Duru Göksu, Gebze Technical University, Computer Engineering Department*

**Project Type:** Deep Learning Research Project — Financial Anomaly Detection

## **1. Introduction**

Financial markets exhibit complex temporal dynamics governed by non-linear interactions, volatility regimes, and abrupt structural shifts. Detecting anomalous behaviour in such systems is challenging due to:

* Heavy-tailed return distributions
* High temporal dependency
* Non-stationary patterns
* Regime changes during crisis periods

This project investigates an **iTransformer-based autoencoder** trained exclusively on *normal market conditions* to learn a robust reconstruction manifold. Deviations from this learned manifold produce measurable increases in reconstruction error (MSE), enabling unsupervised anomaly detection.

The model is evaluated both on **non-crisis data** (AAPL, 2021–2023) and on a **known real-world shock event** (SPY, COVID-19 market crash, 2020). Results show that the reconstruction error correlates strongly with market stress, demonstrating the effectiveness of Transformer-based architectures in financial anomaly detection tasks.

---

## **2. Dataset and Feature Engineering**

### **2.1 Raw Input**

Daily OHLCV data is collected using *Yahoo Finance* for the selected asset.

### **2.2 Derived Features (Total = 9)**

| Feature | Description                       |
| ------- | --------------------------------- |
| Open    | Opening price                     |
| High    | Daily high                        |
| Low     | Daily low                         |
| Close   | Closing price                     |
| Volume  | Total traded volume               |
| HL      | High − Low (intraday range)       |
| OC      | Close − Open                      |
| RET     | Percentage return of close prices |
| VOL_RET | Percentage change in volume       |

These engineered features enable the model to capture price movement, volatility, and liquidity dynamics simultaneously.

### **2.3 Windowing**

A sliding window of length **64 days** is applied:

[
X \in \mathbb{R}^{N \times 64 \times 9}
]

This representation preserves temporal ordering and allows the Transformer to model sequence-level dependencies.

---

## **3. Model Architecture**

### **3.1 iTransformer Autoencoder**

The architecture consists of an encoder–decoder structure built using **iTransformer blocks**, each containing:

* Multi-Head Self-Attention
* Position-wise Feedforward Network
* Layer Normalization layers
* Residual connections
* Dropout regularization

The model projects the input into an embedding space, processes it through several iTransformer blocks, then reconstructs the original sequence.

### **3.2 Objective Function**

The autoencoder is trained using Mean Squared Error:

[
\mathcal{L} = \frac{1}{T \cdot F} \sum (X - \hat{X})^2
]

where

* (T = 64) time steps
* (F = 9) features

The model learns to minimize reconstruction error for normal market sequences.

---

## **4. Training Procedure**

* Dataset: Only windows labelled as **normal** (non-anomalous) are used.
* Optimizer: Adam, learning rate = (1 \times 10^{-4})
* Epochs: 200
* Batch size: 32
* Validation split: 20%
* Callbacks:

  * EarlyStopping (patience = 10)
  * ReduceLROnPlateau
  * ModelCheckpoint (.h5 format for compatibility)

This procedure ensures generalization and prevents overfitting on noise.

---

## **5. Anomaly Detection Methodology**

After inference, window-level anomaly scores are computed as:

[
\text{MSE}_i = \frac{1}{TF} \sum (X_i - \hat{X}_i)^2
]

### **5.1 Thresholding Approaches**

#### **A. Statistical Threshold**

[
T = \mu_{\text{MSE}} + 3\sigma
]

This is suitable for stable datasets but ineffective for crisis data due to heteroskedasticity.

#### **B. Percentile-Based Expert Threshold**

Used for real-market shock periods:

[
T = \text{95th percentile of MSE}
]

This approach is more aligned with financial modelling practices, where distributions are non-Gaussian and volatility clustering is strong.

---

## **6. Real-World Evaluation: SPY 2020 COVID-19 Market Crash**

### **6.1 Motivation**

The SPY crash (Feb–Mar 2020) represents a significant structural break, making it an ideal benchmark for anomaly detection models.

### **6.2 Procedure**

1. Download SPY data from **2019-09 to 2020-07**
2. Apply identical preprocessing pipeline
3. Generate sliding windows (shape: 144 × 64 × 9)
4. Run inference using the trained iTransformer model
5. Compute reconstruction error
6. Apply **95th percentile threshold**

### **6.3 Findings**

* The reconstruction error exhibits a **sharp rise starting at the early decline of SPY**.
* The model detects anomalies consistently at the peak volatility phase.
* Using the percentile method, **four anomaly windows** are correctly identified near the lowest price points.
* The temporal alignment between anomaly spikes and market stress demonstrates:

  * Sensitivity to regime shifts
  * Capability to capture multi-feature deviations
  * Strong generalization to unseen crisis data

---

## **7. Reproducibility Instructions**

### **7.1 Model Loading**

```python
model = keras.models.load_model(
    "itransformer_finance_best.h5",
    custom_objects={"iTransformerBlock": iTransformerBlock}
)
```

### **7.2 Data Preparation**

```python
df = preprocess(df_raw)
windows = create_windows(df.values)
```

### **7.3 Inference**

```python
recon = model.predict(windows)
mse = np.mean((windows - recon)**2, axis=(1, 2))
```

### **7.4 Thresholding**

```python
threshold = np.percentile(mse, 95)
anomalies = mse > threshold
```

---

## **8. Conclusion**

This study demonstrates that the iTransformer autoencoder effectively models normal market behaviour and produces meaningful anomaly scores during abnormal market conditions.

The model successfully detected anomalies in the **2020 COVID-19 crash**, validating its utility for:

* Regime shift detection
* Market crash early-warning indicators
* Multivariate financial anomaly analysis
* Unsupervised risk monitoring

The iTransformer architecture’s ability to capture long-range dependencies makes it a strong candidate for advanced financial time-series modelling.

---

