# Air Quality Prediction — Beijing Multi-Station Dataset

> Multi-step pollutant forecasting using a hybrid ARIMA + CNN-LSTM pipeline across 12 Beijing monitoring stations.

[![Open in Colab

---

## What it does

Predicts next-hour concentrations of 6 key air pollutants — **PM2.5, PM10, SO2, NO2, CO, O3** — from multi-station Beijing air quality data, then maps predictions to CPCB AQI sub-indices for interpretable health-level outputs.

---

## Results

| Metric | Test Score |
|--------|-----------|
| R²     | **0.8637** |
| RMSE   | 0.2726    |
| MAE    | 0.1820    |
| MSE    | 0.0926    |

Model converged at epoch 6 (early stopping at 21); best validation R² = **0.8663**.

---

## Pipeline

```
Raw CSV files (12 stations)
        ↓
Timestamp construction + multi-file merging
        ↓
Missing value imputation
  ├─ Short gaps (<8h): linear interpolation
  └─ Long gaps (≥8h): same-hour prev/next day average
        ↓
Feature engineering
  ├─ Wind direction → degrees → sin/cos cyclical encoding
  ├─ Hour of day → sin/cos cyclical encoding
  └─ MinMaxScaler normalization
        ↓
Feature selection (Decision Tree, per pollutant)
  → Top features: PM2.5, TEMP, DEWP, O3, NO2, wind_sin/cos, hour_cos, CO
        ↓
ARIMA(1,1,1) residuals per pollutant (temporal pattern capture)
        ↓
LSTM dataset: 24-hour sliding window sequences
        ↓
Conv1D → MaxPool → LSTM(64) → Dropout(0.3) → Dense(6)
        ↓
AQI sub-index calculation (CPCB breakpoints)
```

---

## Model Architecture

```
Input (24 timesteps × 17 features)
  → Conv1D (32 filters, kernel=3, relu, same padding)
  → MaxPooling1D (pool=2)
  → LSTM (64 units)
  → Dropout (0.3)
  → Dense (6 outputs — one per pollutant)
```

Trained with Adam (lr=0.0005, clipnorm=1.0), MSE loss, EarlyStopping (patience=15).

---

## Key Design Choices

- **ARIMA residuals as features** — captures temporal autocorrelation that LSTM alone may miss
- **CPCB thresholds** (not generic) used for exceedance classification and AQI mapping
- **Cyclical encoding** for hour and wind direction — prevents the model from treating 23:00→0:00 as a large jump
- **Decision Tree feature selection** per pollutant, respecting CPCB averaging windows (1h NO2, 8h O3/CO, 24h PM/SO2)

---

## Dataset

Beijing Multi-Site Air Quality Dataset — hourly readings from 12 monitoring stations (2013–2017).
Features: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wind direction, wind speed, station ID.

---

## Tech Stack

`Python` `TensorFlow/Keras` `Statsmodels (ARIMA)` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`

---

## Author

**Meghna Gade** — [LinkedIn](https://www.linkedin.com/in/gade-meghna-60a665372/) · IIITDM Jabalpur
