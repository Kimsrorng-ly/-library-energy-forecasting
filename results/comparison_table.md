# Model Comparison & Benchmark Results

All models are evaluated on an identical, strictly chronological 70/30 split of the I-BLEND library smart meter dataset (Feb 2014 – Nov 2017).
- **Target:** Power consumption 24 hours ahead ($t + 144$ steps at 10-minute resolution).
- **Evaluation Rule:** Chronological train/test split (no data leakage). Test set touched exactly once for final reporting.
- **Fair Benchmarking Standard:** All 7 architectures are trained on the unified 27 domain-engineered feature set (`library_featured_v2.csv`).

---

## 1. Grand 7-Model Benchmark (Held-out Test Set)

Evaluated on the exact same 12,375 untouched test readings:

| Rank | Model | Model Family | Test MAE (W) | Test RMSE (W) | Test R² | Train R² | Overfitting Gap (Train - Test) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **XGBoost** | Gradient Boosted Trees | **2,942** | **3,966** | **0.751** | 0.794 | **0.043** |
| 2 | **LightGBM** | Gradient Boosted Trees | 3,087 | 4,150 | 0.727 | 0.867 | 0.139 |
| 3 | **Extra Trees** | Randomized Ensembles | 3,279 | 4,278 | 0.710 | 0.946 | 0.235 |
| 4 | **PyTorch MLP** | Deep Dense Neural Network | 3,469 | 4,636 | 0.660 | 0.828 | 0.168 |
| 5 | **PyTorch BiLSTM** | Bidirectional Recurrent | 3,850 | 5,183 | 0.576 | 0.761 | 0.185 |
| 6 | **PyTorch TCN** | Dilated Causal Convolutional | 4,058 | 5,557 | 0.512 | 0.828 | 0.316 |
| 7 | **PyTorch Transformer** | Multi-Head Self-Attention | 5,030 | 6,887 | 0.251 | 0.850 | 0.599 |

### Baselines for Reference
| Baseline Method | Test MAE (W) | Test RMSE (W) | Test R² | Description |
|---|:---:|:---:|:---:|---|
| **Naive Persistence** | 3,580 | 6,225 | 0.387 | $\hat{y}_{t+24\text{h}} = y_t$ (repeats current value) |
| **Linear Regression** | 3,989 | 5,743 | 0.478 | Standard ordinary least squares on tabular features |

---

## 2. Feature Progression & Ablation Analysis

To evaluate the contribution of domain-informed feature engineering, we compare the tree models across both experimental phases:
- **Phase 1 (8-Feature Baseline):** Trained on `library_cleaned_master.csv` (`power`, `occupancy_count`, `hour`, `dayofweek`, `month`, `weekend`, `power_lag_144`, `power_lag_1008`).
- **Phase 2 (27-Feature Domain Engineering):** Trained on `library_featured_v2.csv` (adding trigonometric cyclical time, BMS campus operational schedules, occupancy dynamics, and rolling statistical moments).

| Model | 8-Feature Test MAE (W) | 8-Feature Test R² | 27-Feature Test MAE (W) | 27-Feature Test R² | Test MAE Reduction | Test R² Gain |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | 3,180 | 0.693 | **2,942** | **0.751** | **-238 W (-7.5%)** | **+0.058 (+8.4%)** |
| **LightGBM** | 3,374 | 0.659 | **3,087** | **0.727** | **-287 W (-8.5%)** | **+0.068 (+10.3%)** |
| **Extra Trees** | 3,271 | 0.680 | **3,279** | **0.710** | **-8 W (-0.2%)** | **+0.030 (+4.4%)** |

> **Key Insight:** Even with the 8-feature baseline, XGBoost ($R^2 = 0.693$) outperformed the best Deep Learning model ($R^2 = 0.660$). When given the full 27 features, XGBoost widens its lead substantially ($R^2 = 0.751$, MAE drops below 3,000 W) with an exceptional generalization gap of only 0.043.

---

## 3. Key Analytical Takeaways

1. **Tabular Feature Engineering vs. Raw Sequences:**
   - Tree models (XGBoost, LightGBM, Extra Trees) and PyTorch MLP achieve strong performance because Notebook 05's engineered features (cyclical trigonometric calendar, university schedule indicators, occupancy dynamics, and 24h/48h autoregressive lags) explicitly provide the temporal context.
   - Deep sequence models (LSTM, TCN) capture momentum and daily diurnal patterns ($R^2 \approx 0.51 - 0.58$), but the lag features already provide this information more compactly and reliably for tree ensembles.

2. **Sample Efficiency & Model Capacity:**
   - The Transformer architecture possesses the highest representational capacity but shows severe overfitting on this single-building dataset ($R^2_{\text{train}} = 0.850$ vs $R^2_{\text{test}} = 0.251$, gap = 0.599).
   - Transformers require significantly larger multi-building datasets or pre-training to compete with boosted trees in building energy forecasting.

3. **Generalization Gap (Overfitting):**
   - XGBoost exhibits the smallest generalization gap ($\Delta R^2 = 0.043$), followed by LightGBM ($\Delta R^2 = 0.139$) and PyTorch MLP ($\Delta R^2 = 0.168$).

---

## 4. Architecture & Hyperparameter Summary

### Tree-Based Models (Trained on 27 Features with TimeSeriesSplit CV)
- **XGBoost:** `n_estimators=300`, `max_depth=4`, `learning_rate=0.01`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_alpha=0.5`, `reg_lambda=3.0`, `tree_method='hist'`
- **LightGBM:** `n_estimators=300`, `num_leaves=31`, `learning_rate=0.01`, `colsample_bytree=0.7`, `subsample=0.8`, `reg_alpha=1.0`, `reg_lambda=2.0`
- **Extra Trees:** `n_estimators=300`, `max_depth=12`, `max_features=1.0`, `min_samples_split=10`, `min_samples_leaf=4`, `bootstrap=True`

### Deep Learning Models (PyTorch Suite, Trained on 27 Features)
- **PyTorch MLP:** 
  - Layers: `Linear(27, 256) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(256, 128) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(128, 64) -> ReLU -> Linear(64, 1)`
  - Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`), StepLR decay (gamma=0.5 every 5 epochs)
- **PyTorch BiLSTM:** 
  - Input: 36-step sequence (6-hour window at 10-min resolution) $\times$ 27 features
  - Architecture: 1-layer Bidirectional LSTM (`hidden_dim=64`, bidir=True $\rightarrow$ output dim 128) + Head: `Linear(128, 64) -> ReLU -> Linear(64, 1)`
- **PyTorch TCN:** 
  - Dilated causal convolutions with dilation factors $d \in \{1, 2, 4\}$, kernel size $k=3$, residual skip connections + Head: `Linear(64, 64) -> ReLU -> Linear(64, 1)`
- **PyTorch Transformer:** 
  - Input Projection: `Linear(27, 64)` + Sinusoidal Positional Encoding
  - Encoder: 2 TransformerEncoderLayers (`d_model=64`, `nhead=4`, `dim_feedforward=128`, `dropout=0.1`)
  - Pooling: Temporal average pooling across sequence $\rightarrow$ Head: `Linear(64, 32) -> ReLU -> Linear(32, 1)`

---

## 5. Experiment Tracking with TensorBoard

All 7 models in `runs/` are logged using the **unified 27-feature set**:
- **`runs/LightGBM`**, **`runs/ExtraTrees`**, **`runs/XGBoost`** (Tree models with 27 features)
- **`runs/MLP`**, **`runs/LSTM`**, **`runs/TCN`**, **`runs/Transformer`** (PyTorch models with 27 features)

To view the fair interactive comparisons, loss curves, computational graphs, and prediction plots:
```bash
tensorboard --logdir=runs/
```
Navigate to `http://localhost:6006` in your browser.
