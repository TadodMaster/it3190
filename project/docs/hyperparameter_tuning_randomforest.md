# Hyperparameter Tuning — Random Forest (DKA-AKI)

Documentation of the hyperparameter-tuning pass applied to the v1 Random Forest
predictor: the methodology, the exact search space, what changed, the measured
results, and how to reproduce. Companion docs: `model_v1_checkpoint.md` (the
baseline this builds on), `data_dictionary.md`, and the method source
`hyper-tuning-random-forest.pdf`.

| | |
|---|---|
| **Task** | Binary classification — predict AKI within 1 week of ICU admission for DKA patients |
| **Model** | `RandomForestClassifier` (scikit-learn 1.1.3) |
| **Notebook** | `src/modeling_randomforest.ipynb` → **Section 7** (new) |
| **Method** | Skill `random-forest-hyperparameter-tuning` (6-step workflow), itself distilled from `hyper-tuning-random-forest.pdf` + scikit-learn docs |
| **Search** | `GridSearchCV`, **1,080 combinations**, Stratified 5-fold CV, scored on ROC-AUC |
| **Logs** | `output/rf_experiment_log.md`, `output/rf_experiment_log.csv`, `output/rf_grid_search_all.csv` |
| **Status** | Done ✔ — CV ROC-AUC **0.8042 → 0.8109**; final model retuned & test predictions regenerated |

---

## 1. Motivation

The v1 checkpoint established a Random Forest baseline (validation AUC ≈ 0.806)
but left hyperparameters at hand-picked defaults (`n_estimators=300`,
`max_depth=None`, `min_samples_leaf=2`). The "next steps" of both the
preprocessing and modeling notebooks called for systematic tuning with k-fold
cross-validation. This pass closes that item and, crucially, **records every
configuration tried** so the choice of final model is auditable.

**Core expectation (from the method):** a Random Forest is already a strong
out-of-the-box model because bagging + feature subsampling de-correlate the trees.
Tuning is expected to give *steady, reliable* gains and better generalization —
not a dramatic jump. The results below confirm exactly that.

---

## 2. Methodology — the 6-step workflow

Implemented in `modeling_randomforest.ipynb` §7, following the
`random-forest-hyperparameter-tuning` skill:

1. **Baseline first.** The current §4 config is measured with the *same* CV used
   for the search and logged as experiment #0, so every later number is comparable.
2. **Grid search.** All combinations of the six meaningful hyperparameters
   (below), exhaustively (`GridSearchCV`).
3. **Cross-validate.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   — stratified so each fold keeps the ~39% positive ratio (vital on imbalanced
   data; a plain split can starve a fold of the minority class).
4. **Metric that matches the task.** Selection is on **ROC-AUC**, not accuracy —
   accuracy is misleading under class imbalance. ROC-AUC is also threshold-
   independent, which separates "is the model better?" from "where do we cut?".
5. **Feature importance** is inspected (§6, unchanged) to confirm the model relies
   on clinically plausible signal (BUN, age, severity scores, eGFR).
6. **Refit on all data.** The winning config is refit on all 970 rows (§8) and
   used for the final test predictions (§9).

`class_weight='balanced'` is held fixed throughout (the imbalance handling chosen
in v1), so the search isolates the effect of the structural hyperparameters.

---

## 3. Search space (all possibilities)

| Parameter | Values tried | Why this range |
|---|---|---|
| `n_estimators` | 200, 300, 500, 800 | more trees = more stable, diminishing returns |
| `max_depth` | None, 5, 10, 20, 30 | primary overfitting control |
| `min_samples_split` | 2, 5, 10 | regularization |
| `min_samples_leaf` | 1, 2, 4 | smoothing / overfitting control |
| `max_features` | 'sqrt', 'log2', 0.5 | controls tree de-correlation |
| `bootstrap` | True, False | bagging on/off |

Total = 4 × 5 × 3 × 3 × 3 × 2 = **1,080 combinations × 5 folds = 5,400 fits**.
Every combination and its CV score is saved to `rf_grid_search_all.csv` (sorted by
rank) — this is the literal "test all possibilities" record.

---

## 4. What changed

| Hyperparameter | Baseline (v1) | Tuned (winner) |
|---|---|---|
| `n_estimators` | 300 | **500** |
| `max_features` | `'sqrt'` | **`'log2'`** |
| `max_depth` | None | None (unchanged) |
| `min_samples_split` | 2 | 2 (unchanged) |
| `min_samples_leaf` | 2 | 2 (unchanged) |
| `bootstrap` | True | True (unchanged) |
| `class_weight` | balanced | balanced (fixed) |

Only two knobs moved. `max_depth=None` (fully grown trees) survived — on this
small dataset the RF's averaging already controls variance, so capping depth did
not help CV AUC. The win came from slightly more trees and a *narrower* feature
sample per split (`log2` < `sqrt` here: log2(77)≈6 vs sqrt(77)≈9 features), which
de-correlates the trees a little more.

> Note: the top three grid rows (`max_depth` ∈ {None, 20, 30}) tie to 4 decimal
> places — depth ≥ 20 is effectively "unlimited" for trees this size. `None` was
> selected as the canonical winner.

---

## 5. Results

### 5.1 Model selection (cross-validated)

| Experiment | CV ROC-AUC | Δ vs baseline |
|---|---|---|
| Baseline (v1 config) | 0.8042 ± 0.0438 | — |
| **GridSearch winner** | **0.8109 ± 0.0445** | **+0.0067** |

A modest but real improvement, exactly as the method predicts for an already-strong
RF. The CV std (~0.044) is larger than the gain, so the honest reading is "the
tuned config is at least as good and slightly better on average," not "a
breakthrough."

### 5.2 Held-out validation (the 194-row split)

| Config | Threshold | AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| Baseline | 0.50 | 0.806 | 0.753 | 0.667 | 0.807 | 0.676 |
| Tuned | 0.50 | 0.807 | 0.758 | 0.667 | 0.815 | 0.680 |
| Tuned | **0.47** (max-F1) | 0.807 | 0.763 | **0.707** | 0.798 | **0.697** |
| Tuned | **0.33** (clinical) | 0.807 | 0.670 | **0.853** | 0.555 | 0.667 |

### 5.3 Threshold analysis

ROC-AUC is unchanged by the threshold (it ranks, it doesn't cut), but the 0/1
labels are not. Because **missing an AKI case (FN) is clinically worse than a false
alarm (FP)**, §7.4 sweeps the threshold and reports two operating points:

- **Max-F1 (0.47):** best precision/recall balance — lifts sensitivity 0.667 →
  0.707 and F1 0.676 → 0.697 at almost no cost to specificity.
- **Clinical (0.33, Sens ≥ 0.85):** catches 85% of AKI cases, at the expected cost
  of specificity (0.555). Use this if the deployment priority is not to miss cases.

This threshold lever moved the practical metrics more than the hyperparameters did
— a key takeaway.

---

## 6. Experiment log (deliverable)

Every experiment above — **including the baseline** — is recorded in three files
under `output/`, generated automatically by §7 of the notebook:

| File | Audience | Contents |
|---|---|---|
| `rf_experiment_log.md` | human | One section per experiment: timestamp, **what changed vs baseline**, full params, CV score, validation metrics, notes |
| `rf_experiment_log.csv` | analysis | Same data as a flat table (one row per experiment) |
| `rf_grid_search_all.csv` | audit | **All 1,080 grid combinations** with `mean_test_score`, `std_test_score`, `rank_test_score` |

The log schema (CSV columns): `timestamp, experiment, changed_vs_baseline,
n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
bootstrap, class_weight, threshold, cv_auc_mean, cv_auc_std, val_AUC,
val_Accuracy, val_Sensitivity, val_Specificity, val_F1, notes`.

---

## 7. Impact on the shipped artifacts

- **Final model** (`output/randomforest_model.joblib`) is now the **tuned** config,
  refit on all 970 rows.
- **Test predictions** (`output/test_predictions_randomforest.csv`) were
  regenerated with the tuned model. The output contract is unchanged: 222
  unique-patient rows, 3 columns (`id`, `probability`, `prediction`), prediction
  at the standard **0.5** threshold. Predicted-AKI rate shifted 40.1% → 42.3%.
  - The alternative thresholds (0.47, 0.33) are documented but **not** applied to
    the shipped file, to honor the original 0.5 spec. Switch by changing the
    threshold in §9 if a different operating point is desired.

---

## 8. How to reproduce

The notebook runs inside the pinned Docker environment (scikit-learn 1.1.3):

```bash
docker compose up -d              # starts it3190-jupyter (JupyterLab :8888)

# Re-run the whole notebook headless (the full grid ≈ 45 min on 6 cores):
docker exec it3190-jupyter jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    /workspace/project/src/modeling_randomforest.ipynb
```

Or open `src/modeling_randomforest.ipynb` in JupyterLab and Run All. Everything is
seeded (`random_state=42`, fixed CV splitter), so results reproduce exactly. The
log files are overwritten on each run (re-initialized in §7.1).

To search a different space, edit `param_grid` in §7.2. To search faster, swap
`GridSearchCV` for `RandomizedSearchCV(n_iter=...)` — see the skill's
`reference/search-strategies.md`.

---

## 9. Conclusions & next steps

**Conclusions**
- Tuning gave a small, reliable CV gain (+0.7 pp AUC) — consistent with RF being a
  strong default. The biggest hyperparameter effect was `max_features='log2'`.
- The **decision threshold** is the more impactful lever for this clinical task and
  is now explicitly tunable, with two recommended operating points logged.
- All 1,080 configurations are on record, so the final choice is fully auditable.

**Next steps**
1. Compare against the tuned XGBoost (`modeling_xgboost.ipynb`) — the paper's best
   model — on the same CV folds and metric.
2. LASSO / importance-based feature selection (paper approach) to cut the 77
   features and possibly reduce variance.
3. Consider probability calibration (`CalibratedClassifierCV`) if the predicted
   probabilities feed a clinical decision rule.
4. Nested CV if an unbiased estimate of the *tuning procedure itself* is needed
   (current numbers slightly favor the selected config by construction).
