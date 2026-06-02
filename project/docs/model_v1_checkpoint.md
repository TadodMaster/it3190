# Model Checkpoint — v1 (Random Forest)

A snapshot of the **first end-to-end training version** of the DKA-AKI predictor:
the reasoning behind each step, the pipeline, the actual results, and what to do
next. Companion docs: `data_dictionary.md` (field definitions) and the source
paper `fpubh-11-1087297.pdf`.

| | |
|---|---|
| **Task** | Binary classification — predict acute kidney injury (AKI) within 1 week of ICU admission for DKA patients |
| **Model** | `RandomForestClassifier` (scikit-learn) |
| **Notebooks** | `src/preprocessing.ipynb` → `src/modeling_randomforest.ipynb` |
| **Artifacts** | `data/train_processed.csv`, `data/preprocessor.joblib`, `output/randomforest_model.joblib`, `output/test_predictions_randomforest.csv` |
| **Status** | Baseline established ✔ — held-out **validation AUC ≈ 0.806** (paper's XGBoost ≈ 0.80) |

---

## 1. Pipeline overview

```
train.json ─► preprocessing.ipynb ─► train_processed.csv (+ preprocessor.joblib)
                                          │
                                          ▼
                              modeling_randomforest.ipynb
                                          │
            train/val split → RF → evaluate → refit on all data
                                          │
test.json ─► (same flatten + preprocessor.transform) ─► predict ─► output CSV
```

Two notebooks, run in order. The preprocessing notebook is the single source of
truth for *how* a patient becomes a feature row; the modeling notebook **reuses**
that exact transform on the test set via the saved `preprocessor.joblib` (no
re-fitting → no data leakage).

---

## 2. Preprocessing — reasoning & steps
(`src/preprocessing.ipynb` → produces a 970 × 77 feature table)

Each record is one ICU stay mixing **static** fields (one value/patient) and
**dynamic** fields (time-series of vitals/labs). Steps:

1. **Flatten to one row per patient.** Dynamic series are aggregated with
   **min, mean, max** (17 series × 3 = 51 columns).
   - *Why min/mean/max and not the paper's "first value":* the extremes carry the
     signal. Example — correlation of `sbp` with the label by statistic:
     **min −0.26, mean +0.05, max +0.27**. A single value (or the mean) hides a
     relationship that min/max expose. Vitals have ~60 readings in 24h, so one
     value wastes most of the data. (See `data_dictionary.md` → "Aggregating
     dynamic features".)
2. **Comorbidity flags: absent ⇒ 0.** The 6 sparse flags (`hypertension`, … ,
   `history_aci`) and `ckd_stage` are recorded *only when present*, so their
   high "missingness" is structural, not missing data — filled with 0/False
   instead of being dropped.
3. **Drop features > 20% missing** (paper rule). With this data **none** of the
   genuinely-measured labs/vitals exceed the threshold (max ≈ 5.9%).
4. **KNN imputation** (`KNNImputer`, k=5) for the remaining numeric gaps.
5. **Encode categoricals:** `gender` F/M→0/1; `liver_disease` NONE/MILD/SEVERE→
   0/1/2 (ordinal); `race` collapsed from 23 raw labels to 5 groups + one-hot;
   `dka_type` (T1DM/T2DM/Other) one-hot.
6. **Standardize** numeric columns (`StandardScaler`). Tree models don't need it,
   but it's harmless and keeps the table model-agnostic.
7. **Save** `train_processed.csv` (970 × 78 incl. label) and `preprocessor.joblib`
   (fitted imputer + scaler + column lists, for reuse on test).

**EDA signal (which features matter):** strongest correlations with AKI were the
severity scores (`saps2`, `oasis` ≈ 0.42; `sofa` ≈ 0.39), `egfr` (≈ −0.42, lower
eGFR → higher risk), `age` (≈ 0.37), `bun_*`, `weight_*`. Group medians (non-AKI →
AKI): `bun_mean` 13.7 → 29.2, `age` 42 → 58, `weight_mean` 68.7 → 81.6,
`egfr` 95.2 → 47.7. This matches the paper's top features (BUN, weight, age).

---

## 3. Modeling — steps
(`src/modeling_randomforest.ipynb`)

1. **Load** `train_processed.csv` → `X` (970 × 77), `y` (38.9% positive).
2. **Stratified split** 80/20 → train 776 / validation 194 (positive rate held at
   ~38.9% / 38.7%). Needed because `test.json` has no labels, so validation is our
   only honest performance estimate.
3. **Train Random Forest** — `n_estimators=300`, `max_depth=None`,
   `min_samples_leaf=2`, `class_weight='balanced'` (for the mild imbalance),
   `random_state=42`.
4. **Evaluate** on validation with the paper's metrics (AUC / accuracy /
   sensitivity / specificity / F1), train-vs-validation comparison, ROC curve,
   feature importance.
5. **Refit on all 970 rows** (`rf_final`) to maximize data before test inference.
6. **Test inference** — reuse `flatten` + `preprocessor.transform`, align to the
   training `feature_cols`, predict, de-duplicate by `subjectId`, save CSV.

---

## 4. Results (v1)

### Validation (held-out 194 patients)
| Metric | Value |
|---|---|
| AUC | **0.806** |
| Accuracy | 0.753 |
| Sensitivity (recall, AKI) | 0.667 |
| Specificity | 0.807 |
| F1 (AKI) | 0.676 |

Confusion matrix `[[TN, FP], [FN, TP]]` = `[[96, 23], [25, 50]]` — i.e. of 75 real
AKI cases, **50 caught / 25 missed**.

### Train vs. Validation (overfitting check)
| Metric | Train | Validation |
|---|---|---|
| AUC | 1.000 | 0.806 |
| Accuracy | 0.997 | 0.753 |
| Sensitivity | 1.000 | 0.667 |
| Specificity | 0.996 | 0.807 |
| F1 | 0.997 | 0.676 |

The near-perfect train scores are expected for an unconstrained Random Forest
(deep trees memorize training rows). The **large train−validation gap signals
overfitting** — the main lever for v2.

### Test inference
- 243 test records → preprocessed to **243 × 77**, columns match train exactly,
  zero missing values.
- **De-duplicated by `subjectId`** (a patient may have several stays with
  different `hadmId`/`stayId`): 243 → **222 unique patients**, keeping the first
  record per patient (21 duplicates dropped).
- Output `output/test_predictions_randomforest.csv` (222 rows): `subjectId`,
  `hadmId`, `stayId`, `akdPositive_proba`, `akdPositive_pred` (threshold 0.5).
  **89 / 222 (40.1%)** predicted AKI — close to the training prevalence (~39%),
  a sanity-check that the distribution looks reasonable.

---

## 5. Key decisions & rationale (quick reference)
| Decision | Choice | Why |
|---|---|---|
| Dynamic-series aggregation | min / mean / max | Extremes are predictive (the `sbp` example); one value wastes 24h of data |
| Sparse comorbidity flags | absent ⇒ 0 | Recorded only when present → not "missing" |
| Missing-data rule | drop > 20%, then KNN-impute | Follows the paper; nothing measured exceeded 20% here |
| Class imbalance | `class_weight='balanced'` + `stratify` split | 38.9% positive — keep ratio honest, weight the minority |
| Test transform | `transform`-only via saved `preprocessor.joblib` | Prevent data leakage |
| Duplicate patients in test | one row per `subjectId` (first) | Project requirement |

---

## 6. Limitations / known issues
- **Overfitting:** train AUC 1.0 vs. validation 0.806 — the model is under-
  regularized. Not yet tuned.
- **Sensitivity 0.667 at threshold 0.5:** misses ~1/3 of AKI cases. Clinically,
  missing AKI (false negative) is worse than a false alarm, so the default 0.5
  threshold is likely sub-optimal.
- **Single split:** validation is one 20% slice; no cross-validation yet, so the
  AUC estimate has variance.
- **First-occurrence de-dup is arbitrary:** for a repeated patient we keep the
  first stay, not (e.g.) the highest-risk one.
- **Feature aggregation differs from the paper** (min/mean/max vs. first value),
  so numbers aren't a 1:1 reproduction — comparable, not identical.

---

## 7. Next steps (v2 candidates)
1. **Reduce overfitting** — `GridSearchCV` over `max_depth` / `min_samples_leaf` /
   `max_features` with stratified k-fold CV (scoring = AUC).
2. **Tune the decision threshold** (e.g. Youden's J) to raise sensitivity.
3. **Try other models** — Logistic Regression and **XGBoost** (the paper's best),
   compare AUC.
4. **Feature selection** (LASSO, as in the paper) to cut noise / dimensionality.
5. Revisit the **de-dup policy** (highest-probability stay instead of first).

> These items are listed as future work in the notebook's final section
> ("Có thể cải tiến"). This checkpoint documents the **v1 baseline** only — the
> current `modeling_randomforest.ipynb` is the untuned Random Forest above.

---

## 8. Reproducibility
- **Run order:** `preprocessing.ipynb` (creates `train_processed.csv` +
  `preprocessor.joblib`), then `modeling_randomforest.ipynb`.
- **Determinism:** `RANDOM_STATE = 42` for the split and both Random Forests.
- **Environment:** the repo's Dockerized JupyterLab (`docker compose up --build`);
  scientific stack pinned in `requirements.txt` (scikit-learn 1.1.3, numpy<2,
  Python 3.11).
- **Inputs:** `data/train.json` (970 labelled), `data/test.json` (243 unlabelled).
