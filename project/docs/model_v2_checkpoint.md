# Model Checkpoint — v2 (XGBoost)

A snapshot of the **second training version** of the DKA-AKI predictor. v1
established a Random Forest baseline; v2 swaps in **XGBoost — the paper's
best-performing model** — on the *same* preprocessing pipeline and the *same*
80/20 split, so the two are directly comparable. Companion docs:
`model_v1_checkpoint.md` (the RF baseline), `data_dictionary.md` (field
definitions), and the source paper `fpubh-11-1087297.pdf`.

| | |
|---|---|
| **Task** | Binary classification — predict acute kidney injury (AKI) within 1 week of ICU admission for DKA patients |
| **Model** | `XGBClassifier` (xgboost 2.1.4) |
| **Notebooks** | `src/preprocessing.ipynb` → `src/modeling_xgboost.ipynb` |
| **Artifacts** | `data/train_processed.csv`, `data/preprocessor.joblib`, `output/xgboost_model.joblib`, `output/test_predictions_xgboost.csv` |
| **Status** | XGBoost iteration ✔ — held-out **validation AUC ≈ 0.817** (beats v1 RF 0.806; paper's XGBoost ≈ 0.800) |

---

## 1. Pipeline overview

```
train.json ─► preprocessing.ipynb ─► train_processed.csv (+ preprocessor.joblib)
                                          │
                                          ▼
                                modeling_xgboost.ipynb
                                          │
            train/val split → XGBoost → evaluate → refit on all data
                                          │
test.json ─► (same flatten + preprocessor.transform) ─► predict ─► output CSV
```

**Identical to v1 except the estimator.** The preprocessing notebook and the
saved `preprocessor.joblib` are unchanged — v2 reuses the exact same feature
table (970 × 77) and the same transform-only test pipeline. Only the modeling
notebook differs (`modeling_xgboost.ipynb` vs. `modeling_randomforest.ipynb`),
which is what makes the AUC numbers comparable across versions.

---

## 2. Preprocessing — unchanged from v1

No changes. v2 consumes the same `train_processed.csv` (970 × 77) and
`preprocessor.joblib` produced by `src/preprocessing.ipynb`. See
`model_v1_checkpoint.md` §2 for the full reasoning (flatten with min/mean/max,
sparse comorbidity flags → 0, drop > 20% missing then KNN-impute, encode
categoricals, standardize, save). Standardization is kept even though trees
don't need it — harmless and keeps the table model-agnostic.

---

## 3. Modeling — steps
(`src/modeling_xgboost.ipynb`)

1. **Load** `train_processed.csv` → `X` (970 × 77), `y` (38.9% positive).
2. **Stratified split** 80/20 → train 776 / validation 194 (positive rate held
   at ~38.9% / 38.7%), `random_state=42` — **same split as v1**, so RF and
   XGBoost are evaluated on the identical 194 patients.
3. **Train XGBoost** — `n_estimators=400`, `learning_rate=0.05`, `max_depth=4`,
   `min_child_weight=2`, `subsample=0.9`, `colsample_bytree=0.9`,
   `reg_lambda=1.0`, `gamma=0.0`, `objective='binary:logistic'`,
   `eval_metric='auc'`, `random_state=42`.
   - **Imbalance:** `scale_pos_weight = n_neg / n_pos = 1.570` (XGBoost's
     equivalent of RF's `class_weight='balanced'`).
4. **Evaluate** on validation with the paper's metrics (AUC / accuracy /
   sensitivity / specificity / F1), train-vs-validation comparison, ROC curve,
   feature importance.
5. **Refit on all 970 rows** (`xgb_final`, `scale_pos_weight` recomputed on the
   full label) to maximize data before test inference.
6. **Test inference** — reuse `flatten` + `preprocessor.transform`, align to the
   training `feature_cols`, predict, de-duplicate by `subjectId`, save CSV.

> **Note on the split vs. the paper:** the paper uses 85/15 + 10-fold CV for
> hyperparameter tuning. We deliberately keep **80/20 stratified** to match v1
> so RF and XGBoost share one validation set. Hyperparameters here are sensible
> hand-picked defaults (small learning rate + shallow trees + subsampling for
> regularization), **not** CV-tuned — that's a v3 lever (§7).

---

## 4. Results (v2)

### Validation (held-out 194 patients)
| Metric | XGBoost (v2) | RF (v1) |
|---|---|---|
| AUC | **0.817** | 0.806 |
| Accuracy | 0.758 | 0.753 |
| Sensitivity (recall, AKI) | **0.720** | 0.667 |
| Specificity | 0.782 | 0.807 |
| F1 (AKI) | **0.697** | 0.676 |

Confusion matrix `[[TN, FP], [FN, TP]]` = `[[93, 26], [21, 54]]` — i.e. of 75
real AKI cases, **54 caught / 21 missed** (v1 caught 50 / missed 25). XGBoost
trades a little specificity (more false alarms: 26 vs. 23) for materially better
sensitivity — the clinically preferable direction, since a missed AKI is worse
than a false alarm.

### Train vs. Validation (overfitting check)
| Metric | Train | Validation |
|---|---|---|
| AUC | 1.000 | 0.817 |
| Accuracy | 1.000 | 0.758 |
| Sensitivity | 1.000 | 0.720 |
| Specificity | 1.000 | 0.782 |
| F1 | 1.000 | 0.697 |

Despite the regularization, on this small dataset **400 trees still memorize the
776 training rows (train AUC ≈ 1.0)** — the train−validation gap is *not* smaller
than v1's. The win over RF is purely on the validation side (higher AUC +
sensitivity), consistent with the paper ranking XGBoost as the best model
(its val AUC 0.800). Closing the gap remains the main lever for v3.

### Feature importance (top 15, XGBoost gain)
`saps2`, `sofa`, `egfr`, `oasis`, `malignant_cancer`, `race_Hispanic`,
`sbp_min`, `macroangiopathy`, `bun_min`, `age`, `weight_min`, `gcs_unable_mean`,
`hb_min`, `dbp_min`, `congestive_heart_failure`.
- Severity scores (`saps2`, `sofa`, `oasis`) and `egfr` dominate — matching the
  EDA correlations in v1 §2.
- Of the paper's top 5 (**BUN, urine output, weight, age, PLT**): `bun_min`,
  `weight_min`, and `age` all appear in our top 10; **urine output is absent
  from this dataset**, and PLT ranks lower here (aggregation differs from the
  paper's single value — see §6 in v1).

### Test inference
- 243 test records → preprocessed to **243 × 77**, columns match train exactly,
  zero missing values.
- **De-duplicated by `subjectId`**: 243 → **222 unique patients** (first record
  per patient, 21 duplicates dropped).
- Output `output/test_predictions_xgboost.csv` (222 rows): `subjectId`,
  `hadmId`, `stayId`, `akdPositive_proba`, `akdPositive_pred` (threshold 0.5).
  **99 / 222 (44.6%)** predicted AKI — somewhat above the training prevalence
  (~39%), in line with XGBoost's higher-sensitivity / lower-threshold behavior.

---

## 5. Key decisions & rationale (what changed vs. v1)
| Decision | v2 choice | Why |
|---|---|---|
| Estimator | `XGBClassifier` | Paper's best of 8 models; gradient boosting corrects errors sequentially |
| Class imbalance | `scale_pos_weight = n_neg/n_pos` (1.570) | XGBoost equivalent of v1's `class_weight='balanced'` |
| Regularization | small `learning_rate` + shallow `max_depth=4` + row/col subsampling + `reg_lambda` | Curb overfitting without CV tuning yet |
| Split | **same** 80/20 stratified, `random_state=42` | Keep RF vs. XGBoost on one validation set |
| Preprocessing / test transform / de-dup | **unchanged from v1** | Isolate the model as the only variable |

---

## 6. Limitations / known issues
- **Still overfitting:** train AUC 1.0 vs. validation 0.817. Regularization
  helped generalization (val AUC ↑) but didn't shrink the gap; hyperparameters
  are hand-picked, not tuned.
- **Sensitivity 0.720 at threshold 0.5:** better than v1 but still misses ~1/4
  of AKI cases; the 0.5 threshold is likely still sub-optimal clinically.
- **Single split:** one 20% validation slice, no cross-validation, so the AUC
  estimate carries variance.
- **First-occurrence de-dup is arbitrary** (carried over from v1).
- **Not a 1:1 paper reproduction:** different feature aggregation
  (min/mean/max vs. first value), different split (80/20 vs. 85/15), no LASSO-CV
  feature selection — comparable, not identical.
- **New dependency:** requires `xgboost==2.1.4` (added to `requirements.txt`);
  the Docker image must be rebuilt for it to be available (see §8).

---

## 7. Next steps (v3 candidates)
1. **Hyperparameter tuning** — `RandomizedSearchCV` / `GridSearchCV` over
   `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`,
   `subsample`, `reg_lambda` with stratified **10-fold CV** (as in the paper),
   scoring = AUC.
2. **Early stopping** on a validation fold (`eval_set` + AUC) to auto-select the
   number of trees and curb the train AUC 1.0 memorization.
3. **Tune the decision threshold** (e.g. Youden's J) to push sensitivity higher.
4. **Feature selection** (LASSO-CV, as in the paper — it screened to 7 variables)
   to cut noise / dimensionality.
5. Revisit the **de-dup policy** (highest-probability stay instead of first).

> These mirror the notebook's final section ("Có thể cải tiến"). This checkpoint
> documents the **untuned XGBoost** in `modeling_xgboost.ipynb` only.

---

## 8. Reproducibility
- **Run order:** `preprocessing.ipynb` (creates `train_processed.csv` +
  `preprocessor.joblib`), then `modeling_xgboost.ipynb`.
- **Determinism:** `RANDOM_STATE = 42` for the split and both XGBoost fits.
- **Environment:** the repo's Dockerized JupyterLab; scientific stack pinned in
  `requirements.txt` (scikit-learn 1.1.3, numpy<2, Python 3.11) **plus
  `xgboost==2.1.4` (newly added)**. Rebuild the image so xgboost is installed:
  `docker compose up -d --build`.
- **Inputs:** `data/train.json` (970 labelled), `data/test.json` (243 unlabelled).
