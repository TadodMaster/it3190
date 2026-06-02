# IT3190 — Nhập môn Học máy và Khai phá dữ liệu

Coursework and final project for **IT3190 – Introduction to Machine Learning and
Data Mining** (semester 2-25-2, class BS25.2B01).

The repository contains hands-on lab/assignment notebooks (data preprocessing and
regression) plus a complete end-to-end final project, all runnable in a
reproducible **Dockerized JupyterLab** environment. Notebook explanations are
written in Vietnamese; supporting documentation (data dictionary, this README) is
in English.

---

## Contents

| Area | Folder | What's inside |
|---|---|---|
| **Final project** | `project/` | DKA-AKI risk prediction on MIMIC-IV — preprocessing + Random Forest, with predictions on a held-out test set |
| **Preprocessing labs** | `preprocessing/` | Tabular data cleaning (Bangalore house prices) and Vietnamese text TF-IDF (VNExpress news) |
| **Regression labs** | `regression/` | Linear & Ridge regression practices + a from-scratch linear regression homework |
| **Clustering labs** | `kmeans/` | K-means with scikit-learn + image compression, and a from-scratch K-means homework |
| **Classification labs** | `knn/` | K-Nearest Neighbors on the German credit dataset (k / feature / distance-metric selection) |
| **Environment** | `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Pinned scientific stack + JupyterLab |

---

## Final project — DKA-AKI prediction (`project/`)

Predict **acute kidney injury (AKI)** within one week of ICU admission for
patients with **diabetic ketoacidosis (DKA)**. A binary classification task on
MIMIC-IV data, following:

> Fan T. *et al.* (2023). *Predicting the risk factors of diabetic
> ketoacidosis-associated acute kidney injury: A machine learning approach using
> XGBoost.* Front. Public Health 11:1087297.

- **Data** (`project/data/`): `train.json` (970 ICU stays, labelled,
  ~38.9% positive) and `test.json` (243 stays, unlabelled). Each record mixes
  **static** features (demographics, comorbidities, severity scores) and
  **dynamic** features (time-series vitals/labs).
- **Pipeline**
  1. `src/preprocessing.ipynb` — flatten each patient to one row (dynamic series
     aggregated with **min/mean/max**), KNN-impute, encode categoricals, scale →
     writes `data/train_processed.csv` and `data/preprocessor.joblib`.
  2. `src/modeling_randomforest.ipynb` — train Random Forest with a stratified
     split, **GridSearchCV tuning** and **decision-threshold tuning**, then apply
     the saved preprocessor to `test.json` → writes
     `output/test_predictions_randomforest.csv`.
- **Docs** (`project/docs/`): the source paper PDF and `data_dictionary.md`
  (every field explained, static vs. dynamic, and the rationale for the
  time-series aggregation choice).

```
project/
├── data/      train.json, test.json  (+ generated: train_processed.csv, preprocessor.joblib)
├── docs/       fpubh-11-1087297.pdf, data_dictionary.md
├── src/        preprocessing.ipynb, modeling_randomforest.ipynb
└── output/     test_predictions_randomforest.csv, randomforest_model.joblib
```

**Run order:** `preprocessing.ipynb` first (creates the processed CSV and the
`preprocessor.joblib`), then `modeling_randomforest.ipynb`.

---

## Coursework

### Preprocessing (`preprocessing/`)
- `practice/data_preprocessing.ipynb` — tabular cleaning on the Bangalore house
  price dataset: missing values, feature engineering, outlier removal, plus a
  self-practice exercise section.
- `assignment/news_vnexpress_/preprocessing_news.ipynb` — Vietnamese news text
  preprocessing: tokenization with `pyvi`, stop-word removal, and TF-IDF
  vectorization over 10 VNExpress topic folders.

### Regression (`regression/`)
- `practice/practice1/` — diabetes dataset: Linear vs. Ridge regression, RMSE.
- `practice/practice2/` — Hyundai Elantra sales: feature scaling, month one-hot
  features, Linear regression.
- `homework/` — Boston housing: a **from-scratch** linear regression (normal
  equation) compared against scikit-learn's implementation.

### Clustering — K-means (`kmeans/`)
- `practice/kmeans.ipynb` — K-means with scikit-learn on synthetic blobs, then
  **image compression** of `bird_small.png` (clustering pixel colours).
- `homework/kmeans_homework.ipynb` — a **from-scratch** `MyKMeans` (k-means++
  init, assignment/update loop, inertia), validated against scikit-learn on both
  the blobs and the image-compression task.

### Classification — KNN (`knn/`)
- `practice/KNN_credit_risk.ipynb` — K-Nearest Neighbors on the Statlog German
  credit dataset with a cost-sensitive evaluation, plus model-selection
  exercises: number of neighbours `k`, feature selection, and distance metric.

> The preprocessing, regression and K-means notebooks were originally written for
> Google Colab and have been localized to run locally (Google Drive mounts and
> `!pip install` cells removed, data paths made relative).

---

## Getting started

### Prerequisites
- [Docker](https://www.docker.com/) (Desktop or Engine) with Compose.

### Launch JupyterLab
```bash
docker compose up --build
```
Then open **http://localhost:8888** (no token/password; the server is bound to
`127.0.0.1` only).

The notebook folders are **bind-mounted**, so edits in the browser are saved
straight back to your working copy.

### Dependencies
Pinned in `requirements.txt`. Notably **`scikit-learn==1.1.3`** (and `numpy<2`,
Python 3.11) — required because `regression_homework.ipynb` uses
`datasets.load_boston()`, removed in scikit-learn 1.2. The Vietnamese tokenizer
`pyvi` is installed with its `sklearn-crfsuite` dependency, and `scikit-image`
provides the image I/O used by the K-means compression notebooks.

---

## Repository layout

```
it3190/
├── preprocessing/   data cleaning + text TF-IDF labs
├── regression/      linear & ridge regression labs + homework
├── kmeans/          K-means clustering + image compression (practice + homework)
├── knn/             K-Nearest Neighbors classification (credit risk)
├── project/         final project: DKA-AKI prediction (MIMIC-IV)
├── Dockerfile       Python 3.11 + JupyterLab image
├── docker-compose.yml
└── requirements.txt pinned scientific stack
```

## License
Released under the [MIT License](LICENSE).
