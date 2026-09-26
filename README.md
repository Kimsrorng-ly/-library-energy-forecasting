# Library Energy Consumption Forecasting

Predicting a university library building's electrical power consumption **24 hours ahead** ($t + 144$ steps at 10-minute resolution) using electrical mains readings and occupancy data from the [I-BLEND dataset](https://doi.org/10.1038/s41597-019-0258-3) (*"I-BLEND, a campus-scale commercial and residential buildings electrical energy dataset"*).

This repository provides an end-to-end Machine Learning and Deep Learning pipeline:
- **3 Tree-Based Ensembles**: LightGBM, Extra Trees, XGBoost
- **4 PyTorch Deep Learning Models**: MLP, BiLSTM, Temporal Convolutional Network (TCN), and Transformer
- **Centralized Experiment Tracking**: Real-time TensorBoard logging (scalars, loss curves, computational graphs, prediction time-series, and hyperparameter tables)

---

## 1. Repository Structure

```
.
├── notebooks/
│   ├── 01_data_cleaning.ipynb          # Sensor gap alignment, interpolation, downsampling (10-min)
│   ├── 02_lightgbm.ipynb               # Baseline: LightGBM regressor with TimeSeriesSplit tuning
│   ├── 03_extratrees.ipynb             # Baseline: Extra Trees regressor
│   ├── 04_xgboost.ipynb                # Baseline: XGBoost regressor (best tree model)
│   ├── 05_feature_engineering_v2.ipynb # 27 domain-aware features (cyclical time, calendar, occupancy)
│   ├── 06_mlp_pytorch.ipynb            # Deep Dense MLP (PyTorch)
│   ├── 07_lstm_pytorch.ipynb           # Bidirectional LSTM sequence model (PyTorch)
│   ├── 08_tcn_pytorch.ipynb            # Dilated Causal Temporal Convolutional Network (PyTorch)
│   └── 09_transformer_pytorch.ipynb    # Multi-Head Self-Attention Transformer (PyTorch)
├── utils/
│   ├── __init__.py                     # Package export
│   └── tb_logger.py                    # Unified SklearnTBLogger and TorchTBLogger for TensorBoard
├── results/
│   └── comparison_table.md             # Detailed benchmark report across all 7 models
├── runs/                               # TensorBoard event runs for all models
├── library_cleaned_master.csv          # Cleaned 10-min dataset (118,776 rows)
├── LB.csv                              # Raw occupancy data
├── requirements.txt                    # Project dependencies
└── README.md
```

---

## 2. Pipeline & Workflow

```
[Raw Data: LB.csv & mains]
           │
           ▼
[01_data_cleaning.ipynb] ──► library_cleaned_master.csv (118,776 rows)
           │
   ┌───────┴────────────────────────────────────────┐
   ▼                                                ▼
[Phase 1: Exploratory 8-Feature Baselines]  [05_feature_engineering_v2.ipynb]
(LightGBM / Extra Trees / XGBoost)                  │
                                                    ▼
                                          library_featured_v2.csv (41,248 rows, 27 features)
                                                    │
                                  ┌─────────────────┼─────────────────┐
                                  ▼                 ▼                 ▼
                       [Phase 2: Tree Models]   [06 MLP]       [07 BiLSTM]
                       (XGB / LGBM / ET on 27F)     │                 │
                                                    ▼                 ▼
                                                [08 TCN]      [09 Transformer]
                                                    │
                                                    ▼
                                         [TensorBoard Dashboard]
                                        (All 7 Models on 27 Features)
```

### Step 1: Data Cleaning (`notebooks/01_data_cleaning.ipynb`)
- Converts timestamps to `Asia/Kolkata` local time.
- Reconstructs missing electrical `current` via $I = P / (V \times PF)$ and corrects power factor anomalies.
- Exposes hidden sensor gaps on a 1-minute grid; interpolates only short dropouts ($\le 10$ minutes) and leaves long blackouts as `NaN`.
- Downsamples to 10-minute resolution to match native occupancy recording.
- Merges power and occupancy, producing `library_cleaned_master.csv`.

### Step 2: Advanced Feature Engineering (`notebooks/05_feature_engineering_v2.ipynb`)
Generates 27 continuous, domain-informed features specifically designed for Building Management Systems (BMS):
- **Cyclical Temporal**: `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `month_sin`, `month_cos` (removes midnight/weekend boundary discontinuities).
- **Academic Timetable**: `is_class_hour` (8:00–18:00 weekdays), `is_evening_study` (18:00–22:00), `is_night`, `is_exam_month` (Apr, May, Nov, Dec), `is_vacation_month` (Jun, Jul).
- **Occupancy Dynamics**: 1-hour rolling mean, velocity/delta, and estimated active zones (`rooms_occupied_est`).
- **Autoregressive Lags & Volatility**: 24h lag (`power_lag_144`), 48h lag (`power_lag_288`), 1-week lag (`power_lag_1008`), and 24h rolling standard deviation.

### Step 3: Model Training & Evaluation
All models adhere to a **strict chronological 70/30 train/test split**:
- **Train period**: 28,873 rows (Feb 2014 – Nov 2016)
- **Test period**: 12,375 rows (Nov 2016 – Nov 2017)
- Target: Unconstrained linear output for power prediction in Watts (never classification / softmax).

---

## 3. Benchmark Results

### A. Fair 7-Model Benchmark (All Models on Unified 27 Features)
Evaluated on the exact same 12,375 untouched test readings:

| Rank | Model | Model Family | Test MAE (W) | Test RMSE (W) | Test R² | Train R² | Overfitting Gap |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **XGBoost** | Gradient Boosted Trees | **2,942** | **3,966** | **0.751** | 0.794 | **0.043** |
| 2 | **LightGBM** | Gradient Boosted Trees | 3,087 | 4,150 | 0.727 | 0.867 | 0.139 |
| 3 | **Extra Trees** | Randomized Ensembles | 3,279 | 4,278 | 0.710 | 0.946 | 0.235 |
| 4 | **PyTorch MLP** | Deep Dense Neural Network | 3,469 | 4,636 | 0.660 | 0.828 | 0.168 |
| 5 | **PyTorch BiLSTM** | Bidirectional Recurrent | 3,850 | 5,183 | 0.576 | 0.761 | 0.185 |
| 6 | **PyTorch TCN** | Dilated Causal Convolutional | 4,058 | 5,557 | 0.512 | 0.828 | 0.316 |
| 7 | **PyTorch Transformer** | Multi-Head Self-Attention | 5,030 | 6,887 | 0.251 | 0.850 | 0.599 |

*Baselines: Naive Persistence ($R^2 = 0.387$, MAE = 3,580 W), Linear Regression ($R^2 = 0.478$, MAE = 3,989 W).*

### B. Feature Progression & Ablation (8-Feature Baseline vs. 27-Feature Domain Engineering)
Demonstrating the direct impact of domain-informed features on tree ensemble performance:

| Model | 8-Feature Test MAE (W) | 8-Feature Test R² | 27-Feature Test MAE (W) | 27-Feature Test R² | Test MAE Gain | Test R² Gain |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | 3,180 | 0.693 | **2,942** | **0.751** | **-238 W (-7.5%)** | **+0.058 (+8.4%)** |
| **LightGBM** | 3,374 | 0.659 | **3,087** | **0.727** | **-287 W (-8.5%)** | **+0.068 (+10.3%)** |
| **Extra Trees** | 3,271 | 0.680 | **3,279** | **0.710** | **-8 W (-0.2%)** | **+0.030 (+4.4%)** |

---

## 4. TensorBoard Experiment Tracking

All training and evaluation runs are logged using `utils/tb_logger.py`:

```bash
tensorboard --logdir=runs/
```
Open **`http://localhost:6006`** to inspect:
- **`HPARAMS`**: Consolidated comparison table of all 7 models sorted by Test MAE and Test R².
- **`GRAPHS`**: Computational graphs showing model connectivity down to the `Linear(1)` regression head.
- **`SCALARS`**: Epoch-by-epoch training and validation loss curves and learning rate schedules.
- **`IMAGES`**: Actual vs. Predicted 24-hour time series overlays and residual error distributions.

---

## 5. Quickstart

### 1. Installation
```bash
# Clone the repository
git clone <repo-url>
cd -library-energy-forecasting

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline
Open Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook
```
Execute notebooks in sequence:
1. `notebooks/01_data_cleaning.ipynb` *(optional if `library_cleaned_master.csv` is already present)*
2. `notebooks/02_lightgbm.ipynb`, `03_extratrees.ipynb`, `04_xgboost.ipynb`
3. `notebooks/05_feature_engineering_v2.ipynb` *(creates `library_featured_v2.csv`)*
4. `notebooks/06_mlp_pytorch.ipynb`, `07_lstm_pytorch.ipynb`, `08_tcn_pytorch.ipynb`, `09_transformer_pytorch.ipynb`
