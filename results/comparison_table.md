# Model Comparison — Full Results

All models trained and evaluated on the identical pipeline described in the main
[README](../README.md): `library_cleaned_master.csv` → gap-aware feature engineering → 70/30
chronological split (28,873 train / 12,375 test) → `RandomizedSearchCV` tuning with
`TimeSeriesSplit(gap=144)` → single evaluation on the held-out test set.

## Final test-set performance

| Model | MAE (W) | RMSE (W) | R² |
|---|---|---|---|
| Naive Persistence | 3580 | 6225 | 0.387 |
| Linear Regression | 3989 | 5743 | 0.478 |
| LightGBM | 3374 | 4643 | 0.659 |
| Extra Trees | 3271 | 4495 | 0.680 |
| XGBoost | 3180 | 4405 | 0.693 |

## Train vs. test (overfitting check)

| Model | Train R² | Test R² | R² gap |
|---|---|---|---|
| LightGBM | 0.817 | 0.659 | 0.158 |
| Extra Trees | 0.809 | 0.680 | 0.129 |
| XGBoost | 0.793 | 0.693 | 0.100 |

XGBoost shows the smallest train/test gap of the three, suggesting the best generalization among
these models on this dataset, despite all three showing some expected overfitting relative to
their training performance.

## 5-fold TimeSeriesSplit cross-validation (training set only, gap=144)

**Extra Trees**

| Fold | MAE | RMSE | R² |
|---|---|---|---|
| 1 | 5268 | 8025 | 0.086 |
| 2 | 4265 | 5596 | 0.364 |
| 3 | 3059 | 4337 | 0.491 |
| 4 | 2463 | 3047 | 0.760 |
| 5 | 2970 | 4408 | 0.651 |
| **Mean** | **3605** | **5082** | **0.470** |
| Std | 1021 | 1678 | 0.235 |

**XGBoost**

| Fold | MAE | RMSE | R² |
|---|---|---|---|
| 1 | 5603 | 8492 | -0.023 |
| 2 | 4276 | 5573 | 0.369 |
| 3 | 2794 | 3971 | 0.573 |
| 4 | 2475 | 3148 | 0.744 |
| 5 | 2950 | 4224 | 0.679 |
| **Mean** | **3620** | **5081** | **0.469** |
| Std | 1167 | 1875 | 0.277 |

Early folds score lower for both models since they train on less history — expected behavior for
`TimeSeriesSplit`, not a red flag.

## Best hyperparameters found (RandomizedSearchCV, n_iter=20)

**Extra Trees**
```
n_estimators: 800
max_depth: 10
max_features: 0.5
min_samples_split: 20
min_samples_leaf: 10
bootstrap: True
```
Best CV R² (3-fold, tuning only): 0.601

**XGBoost**
```
n_estimators: 300
max_depth: 5
learning_rate: 0.01
subsample: 0.7
colsample_bytree: 0.7
min_child_weight: 5
gamma: 0.1
reg_alpha: 0.5
reg_lambda: 3.0
```
Best CV R² (3-fold, tuning only): 0.606

*(LightGBM's tuned parameters are in `notebooks/02_lightgbm.ipynb`, Section 8.)*
