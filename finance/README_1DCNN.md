# Financial Time-Series Anomaly Detection Using a 1D Convolutional Autoencoder

## Abstract

This repository presents an unsupervised anomaly detection framework for financial time-series data using a one-dimensional Convolutional Autoencoder (1D-CNN AE). The model is trained exclusively on historical data representing normal market behavior and is evaluated on previously unseen time periods to assess its ability to detect structural changes, volatility shifts, and deviations from typical price dynamics. Reconstruction error is used as the primary anomaly score. The entire pipeline, including data acquisition, preprocessing, feature engineering, model training, and testing on unseen data, is fully documented.

---

## 1. Introduction

Detecting anomalous behavior in financial markets is essential for understanding structural changes, volatility regime shifts, and disruptive price events. However, financial datasets rarely contain explicit anomaly labels. This motivates unsupervised learning approaches capable of identifying deviations solely from historical normal behavior.

In this project, a 1D Convolutional Autoencoder is employed to model normal temporal patterns observed in a stock price time series. The model is trained on a multi-year period presumed to reflect typical behavior and then evaluated on subsequent years to determine whether reconstruction error can serve as an effective indicator of abnormal market conditions.

---

## 2. Dataset and Preprocessing

### 2.1 Data Source

Financial price data is obtained through the `yfinance` API.

* **Training period:** 2015–2020
* **Testing period:** 2021–2023
* **Asset:** Apple Inc. (AAPL)

### 2.2 Selected Features

The following six fundamental features are used:

* Open
* High
* Low
* Close
* Volume
* Adjusted Close (constructed as `Close`, due to `auto_adjust=True` in yfinance)

All features are normalized using `StandardScaler`.

### 2.3 Sliding Window Formation

A fixed-length sliding window of **64 time steps** is applied to construct model inputs. This ensures local temporal structure is captured and maintains consistency between the training and testing pipelines.

### 2.4 Feature Engineering

To enrich the representation and provide the autoencoder with additional temporal signals, three engineered features are appended to each window:

1. Volatility Change Rate
2. Rolling Mean Difference
3. Price Jump Ratio

This results in a **9-dimensional** input representation per time step.

---

## 3. Model Architecture

### 3.1 1D Convolutional Autoencoder

The model consists of an encoder that compresses local temporal patterns and a decoder that reconstructs them.

**Encoder**

* Conv1D with 128 filters (kernel size 7)
* MaxPooling1D
* Conv1D with 64 filters (kernel size 5)
* MaxPooling1D

**Decoder**

* Conv1D
* UpSampling1D
* Conv1D output layer (channels = 9)

### 3.2 Training Procedure

* Loss function: Mean Squared Error (MSE)
* Optimizer: Adam (learning rate 1e-4)
* Training data: All windows derived from the 2015–2020 period
* Validation split: 20%
* Early stopping and learning rate scheduling applied

The model learns to reconstruct normal patterns present in historical financial sequences.

---

## 4. Testing Phase on Unseen Data (2021–2023)

### 4.1 Objective

The model is evaluated on an unseen period (2021–2023) to assess its ability to detect deviations from previously learned patterns. Since no anomaly labels exist for financial price data, the evaluation is carried out through reconstruction error analysis and alignment with observable market movements.

### 4.2 Testing Pipeline

The testing process mirrors the training pipeline to ensure methodological consistency:

1. Fetch raw price data for the test period.
2. Apply identical preprocessing steps (feature selection, scaling, windowing).
3. Apply the same feature engineering transformations used during training.
4. Generate inputs of shape `(num_samples, 64, 9)`.
5. Load the trained 1D-CNN Autoencoder.
6. Compute reconstruction error for each window:
   [
   \mathrm{MSE} = \frac{1}{T \cdot F} \sum (x - \hat{x})^2
   ]
7. Visualize reconstruction error relative to time and compare it to contemporaneous price movements.

### 4.3 Observations

The reconstruction error reveals several notable properties:

* **Stable periods** in price dynamics correspond to consistently low error.
* **Areas of structural change**, such as trend reversals and volatility spikes, consistently produce elevated reconstruction error.
* **High-magnitude price fluctuations** coincide with abrupt increases in reconstruction error, indicating that the model identifies these as deviations from learned normal patterns.

This demonstrates the utility of the autoencoder as a **regime shift detector** in financial time-series analysis.

---

## 5. Usage Instructions

### Installation

```
pip install -r requirements.txt
```

### Run Preprocessing

```
python data_preprocessing.py
```

### Train the Autoencoder

```
python train_autoencoder.py
```

### Evaluate on Unseen Data

```
python test_autoencoder.py
```

---

## 6. Repository Structure

A recommended structure for this repository is as follows:

```
/models                     # Trained Autoencoder, Hybrid Models
/data                       # Raw, processed, and windowed datasets
/notebooks                  # Jupyter/Colab notebooks for training and testing
/reports                    # Automatically generated evaluation reports
train_autoencoder.py
test_autoencoder.py
README.md
```

---

## 7. Significance of the Approach

This methodology provides a systematic unsupervised mechanism for analyzing financial markets without reliance on labeled anomalies. Reconstruction error serves as an effective indicator of:

* Volatility regime shifts
* Trend discontinuities
* Structural breaks in market behavior
* Temporal pattern deviations

Such systems are valuable for quantitative risk management, early warning systems, and as a feature generator for hybrid supervised approaches.

---

## 8. Future Work

Potential directions for extending this work include:

* Comparative evaluation using LSTM Autoencoders
* Experimentation with Transformer-based Autoencoders (e.g., iTransformer)
* Statistical calibration of anomaly thresholds
* Hybrid methods integrating autoencoder scores with gradient-boosting models
* Event-driven interpretation using external financial news sources

---

## 9. Author

Bengisu Göksu
Gebze Technical University

---

