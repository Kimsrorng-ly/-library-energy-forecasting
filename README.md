# Library Energy Consumption Forecasting

Predicting a university library building's power consumption **24 hours ahead**, using electrical
mains data and occupancy counts from the [I-BLEND dataset](https://doi.org/10.1038/s41597-019-0258-3)
("I-BLEND, a campus-scale commercial and residential buildings electrical energy dataset").

Three tree-based models — **LightGBM**, **Extra Trees**, and **XGBoost** — are trained on an
identical data pipeline and compared head-to-head, so any difference in results reflects the
models themselves rather than the data setup.

## Data

Only the **library building's** mains meter and occupancy readings are used from I-BLEND (the
original dataset covers a full campus of buildings):

| File | Description | Rows | Interval |
|---|---|---|---|
| `library_build_mains.csv` | Raw electrical mains readings (power, current, voltage, frequency, power factor) | 1,713,637 | 1 minute |
| `LB.csv` | Raw occupancy counts | 173,006 | 10 minutes |
| `library_cleaned_master.csv` | Cleaned, merged output of the notebook below | 118,776 | 10 minutes |

> Raw/intermediate CSVs are large — see [Data files](#data-files) below for how they're handled in
> this repo.

## Pipeline

### 1. Data cleaning — [`notebooks/01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb)
- Converts Unix timestamps to `Asia/Kolkata` local time
- Reconstructs missing `current` via `I = P / (V × PF)`, corrects sign anomalies in `power_factor`
- Exposes hidden sensor gaps by reindexing onto a complete 1-minute time grid
- Interpolates only short gaps (≤10 minutes); leaves longer blackouts as `NaN`
- Downsamples to 10-minute resolution (mean) to match the occupancy data's native interval
- Merges mains + occupancy on timestamp, drops remaining blackout rows
- Output: `library_cleaned_master.csv` (118,776 rows, Feb 2014 – Nov 2017)

### 2. Feature engineering (repeated identically in each model notebook)
- Calendar features: `hour`, `dayofweek`, `month`, `day_of_year`, `weekend`
- **Gap-aware** lag features (`power_lag_144` = same time yesterday, `power_lag_1008` = same time
  last week) — a plain `.shift()` would silently pull values from across a sensor gap and mislabel
  them; each lag is checked against the rolling max of `time_diff` and set to `NaN` if a gap falls
  inside the window
- Backward-looking rolling means (`power_roll_mean_144`, `power_roll_mean_1008`) via `.shift(1)`
  before `.rolling()`, guaranteeing no future leakage
- Target: `power` shifted 144 steps into the future (24 hours ahead)
- `current`, `voltage`, `frequency`, `power_factor` are **excluded** as model inputs — `power`
  correlates with `current` at r ≈ 0.99 (near-restatement of the target), `current` was partially
  reconstructed from `power` during cleaning (leakage risk), and none of these instantaneous
  readings would be available at prediction time for a real 24h-ahead forecast anyway
- After dropping incomplete rows (insufficient history, gap-invalidated features, undefined
  target): **41,248 rows**

### 3. Train/test split
Chronological 70/30 split, no shuffling (shuffling a time series before splitting leaks future
data into training):
- Train: 28,873 rows (2014-02-26 → 2016-11-20)
- Test: 12,375 rows (2016-11-20 → 2017-11-02)

### 4. Models — [`notebooks/02_lightgbm.ipynb`](notebooks/02_lightgbm.ipynb), [`03_extratrees.ipynb`](notebooks/03_extratrees.ipynb), [`04_xgboost.ipynb`](notebooks/04_xgboost.ipynb)
Each notebook:
- Runs the identical feature engineering + split above
- Establishes two baselines: naive persistence (`power(t+24h) = power(t)`) and Linear Regression
- Tunes hyperparameters via `RandomizedSearchCV` (`n_iter=20`) with `TimeSeriesSplit(gap=144)`
- Reports 5-fold `TimeSeriesSplit(gap=144)` cross-validation on the training set only
- Fits a final model and evaluates on the untouched test set exactly once

Model-specific adjustments (each justified in that notebook's own markdown):
- **LightGBM**: early stopping on a held-out 10% slice of the training set; `n_jobs=1` to avoid
  Colab RAM issues during parallel boosting
- **Extra Trees**: no early stopping (no boosting sequence to interrupt) — regularization instead
  comes from tuned `max_depth`/`min_samples_leaf`/`min_samples_split`; `n_jobs=-1` since Extra
  Trees fits are lighter and embarrassingly parallel
- **XGBoost**: early stopping via `eval_set`, similar to LightGBM

## Results

Final evaluation on the held-out test set (touched once per notebook):

| Model | MAE (W) | RMSE (W) | R² |
|---|---|---|---|
| Naive Persistence (baseline) | 3580 | 6225 | 0.387 |
| Linear Regression (baseline) | 3989 | 5743 | 0.478 |
| LightGBM | 3374 | 4643 | 0.659 |
| Extra Trees | 3271 | 4495 | 0.680 |
| **XGBoost** | **3180** | **4405** | **0.693** |

All three models clearly outperform both baselines. XGBoost edges out Extra Trees and LightGBM on
every metric, though the margins between the three tree-based models are modest compared to the
gap over the baselines.

See [`results/comparison_table.md`](results/comparison_table.md) for the full table including
cross-validation and train-set metrics.

## Limitations

- **No weather data** — temperature is a major driver of HVAC load and isn't available here
- **No holiday calendar** — models can't distinguish holidays/breaks from regular weekdays
- **Static models** — trained once; real deployment would need periodic retraining
- **Sensor gaps** — 1,124 gaps >10 minutes in the raw mains data; gap-aware feature engineering
  prevents corrupted lags but rows near a gap still carry less complete history
- **24-hour horizon only** — a full next-24-hours trajectory would need recursive/multi-output
  forecasting, which is outside this project's scope
- **Concept drift** — average power consumption shifts across the multi-year span (see the
  cleaning notebook's EDA); tree-based models can't extrapolate beyond the range of `power` values
  seen in training, worth watching for in future work

## Data files

`library_build_mains.csv` (~93 MB) is excluded from version control via `.gitignore` since it
exceeds comfortable git repo size. To reproduce this project from scratch:
1. Download the I-BLEND dataset's library building mains and occupancy files
2. Place them in the repo root as `library_build_mains.csv` and `LB.csv`
3. Run `notebooks/01_data_cleaning.ipynb` to regenerate `library_cleaned_master.csv`

`library_cleaned_master.csv` (~9 MB) and `LB.csv` are small enough to commit directly.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order: `01_data_cleaning.ipynb` → any of `02`/`03`/`04` (each is
self-contained and only depends on `library_cleaned_master.csv`).
