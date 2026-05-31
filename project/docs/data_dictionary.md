# Data Dictionary — DKA-AKI Dataset

Dataset for predicting **diabetic ketoacidosis–associated acute kidney injury
(DKA-AKI)** from MIMIC-IV, following Fan et al. 2023 (`fpubh-11-1087297.pdf`).
Each record is one ICU stay of a DKA patient; the task is to predict whether the
patient develops AKI within one week of ICU admission.

## Files & overview
| File | Records | Has label? |
|---|---|---|
| `train.json` | 970 | yes (`akdPositive`) |
| `test.json`  | 243 | no |

Label balance in train: **377 positive (38.9%) / 593 negative**.

## Static vs. dynamic (tĩnh vs. động)
- **Static (tĩnh)** — a single value per patient (demographics, comorbidities,
  severity scores). Stored as a scalar in `measures`.
- **Dynamic (động)** — a *time series*: a `{timestamp: value}` map with one or
  more measurements over the first 24h in ICU. Stored as a nested object.
  The paper keeps **only the first measurement** of each dynamic feature.

All measurements share the same JSON shape: `measures` is an object whose keys
are the field names below.

## Top-level fields
| Field | Type (tĩnh/động) | Data type | Description |
|---|---|---|---|
| `subjectId` | static (tĩnh) | int | Mã định danh bệnh nhân (MIMIC `subject_id`). |
| `hadmId` | static (tĩnh) | int | Mã lần nhập viện (`hadm_id`). |
| `stayId` | static (tĩnh) | int | Mã lần nằm ICU (`stay_id`). |
| `akdPositive` | static (tĩnh) | bool | **Nhãn mục tiêu.** AKI trong vòng 1 tuần sau khi nhập ICU (theo tiêu chuẩn KDIGO). Chỉ có ở tập train. |
| `measures` | — | object | Đối tượng chứa toàn bộ các trường đặc trưng liệt kê bên dưới. |

## Static features (tĩnh) — one value per patient
| Field | Data type | Present % | Description |
|---|---|---|---|
| `age` | int | 100% | Tuổi (năm). |
| `gender` | str (`F`/`M`) | 100% | Giới tính. |
| `race` | str | 100% | Chủng tộc; 23 nhãn gốc của MIMIC (vd `WHITE`, `BLACK/AFRICAN AMERICAN`, `HISPANIC...`, `ASIAN...`, `UNKNOWN`). Thường gộp thành White/Black/Hispanic/Asian/Other. |
| `dka_type` | int | 100% | Loại đái tháo đường: `1`=T1DM, `2`=T2DM, `0`=Other. (Số lượng 600/275/95 khớp với tỉ lệ 62%/28%/10% trong bài báo.) |
| `liver_disease` | str | 100% | Mức độ bệnh gan: `NONE`/`MILD`/`SEVERE` (có thứ tự). |
| `oasis` | float | 100% | Điểm độ nặng OASIS. |
| `saps2` | int | 100% | Điểm độ nặng SAPS II. |
| `sofa` | int | 100% | Điểm suy tạng SOFA. |
| `preiculos` | float | 100% | Thời gian nằm viện trước khi vào ICU (pre-ICU length of stay, đơn vị theo MIMIC). |
| `egfr` | float | 99.5% | Mức lọc cầu thận ước tính (eGFR — chức năng thận). Ở đây là một giá trị đơn. |
| `chronic_pulmonary_disease` | bool | 100% | Cờ bệnh nền (`True`/`False` rõ ràng) — bệnh phổi mạn tính. |
| `congestive_heart_failure` | bool | 100% | Cờ bệnh nền (`True`/`False`) — suy tim sung huyết. |
| `malignant_cancer` | bool | 100% | Cờ bệnh nền (`True`/`False`) — ung thư ác tính. |
| `hypertension` | bool | 39.0% | Cờ bệnh nền (tăng huyết áp) — **chỉ xuất hiện khi True**; vắng mặt ⇒ không mắc (coi là `False`/0). |
| `microangiopathy` | bool | 36.7% | Biến chứng mạch máu nhỏ do đái tháo đường — chỉ xuất hiện khi True ⇒ vắng mặt = `False`. |
| `macroangiopathy` | bool | 23.1% | Biến chứng mạch máu lớn do đái tháo đường — chỉ xuất hiện khi True ⇒ vắng mặt = `False`. |
| `ckd_stage` | int | 20.2% | Giai đoạn bệnh thận mạn (CKD; quan sát thấy 1–4); vắng mặt ⇒ không CKD (coi là `0`). |
| `history_ami` | bool | 15.1% | Tiền sử nhồi máu cơ tim cấp (acute myocardial infarction) — chỉ xuất hiện khi True ⇒ vắng mặt = `False`. |
| `uti` | bool | 12.1% | Nhiễm trùng đường tiết niệu (urinary tract infection) — chỉ xuất hiện khi True ⇒ vắng mặt = `False`. |
| `history_aci` | bool | 7.2% | Tiền sử nhồi máu não cấp (acute cerebral infarction / đột quỵ) — chỉ xuất hiện khi True ⇒ vắng mặt = `False`. *(Viết tắt mang tính suy đoán; cần đối chiếu nguồn.)* |

> **Note on the sparse comorbidity flags** (`hypertension` … `history_aci`, and
> `ckd_stage`): their low "present %" is **structural, not missing data** — the
> field is recorded only when the condition exists (every recorded value is
> `True`). So absence means "condition not present" and should be filled with
> `0`/`False`, **not** dropped by a missing-rate threshold. Only the measured
> labs/vitals below are genuinely subject to missingness.

## Dynamic features (động) — time series `{timestamp: value}`
Timestamps are ISO-8601 strings (e.g. `2133-01-25T04:39:00`), so lexical order =
chronological order. Per-record measurement counts vary widely (see "Timepoints").

| Field | Đơn vị / ý nghĩa | Present % | Timepoints (min/median/max) |
|---|---|---|---|
| `hr` | Nhịp tim (heart rate, beats/min) | 99.9% | 1 / 40 / 1119 |
| `rr` | Nhịp thở (respiratory rate, breaths/min) | 99.9% | 1 / 39 / 1130 |
| `sbp` | Huyết áp tâm thu (systolic BP, mmHg) | 99.7% | 1 / 35 / 1133 |
| `dbp` | Huyết áp tâm trương (diastolic BP, mmHg) | 99.7% | 1 / 35 / 1133 |
| `gcs` | Thang điểm hôn mê Glasgow (GCS, tổng) | 99.8% | 1 / 5 / 27 |
| `gcs_unable` | Cờ "không đánh giá được" GCS (0/1) | 99.8% | 1 / 5 / 27 |
| `ag` | Khoảng trống anion (anion gap, mmol/L) | 99.5% | 1 / 4 / 14 |
| `bg` | Đường huyết (blood glucose, mg/dL) | 99.5% | 1 / 4 / 14 |
| `bicarbonate` | Bicarbonate huyết thanh (mmol/L) | 99.5% | 1 / 4 / 14 |
| `bun` | Nitơ urê máu (BUN, mg/dL) | 99.5% | 1 / 4 / 14 |
| `scr` | Creatinine huyết thanh (serum creatinine, mg/dL) | 99.5% | 1 / 4 / 14 |
| `phosphate` | Phosphate huyết thanh (mg/dL) | 99.1% | 1 / 4 / 13 |
| `calcium` | Canxi huyết thanh (mg/dL) | 99.0% | 1 / 4 / 14 |
| `weight` | Cân nặng (kg) | 98.4% | 1 / 1 / 1 |
| `plt` | Số lượng tiểu cầu (platelet count, 10^9/L) | 94.3% | 1 / 2 / 10 |
| `hb` | Hemoglobin (g/dL) | 94.1% | 1 / 2 / 10 |
| `wbc` | Số lượng bạch cầu (WBC, 10^9/L) | 94.1% | 1 / 2 / 10 |

> Vital signs (`hr`, `rr`, `sbp`, `dbp`, `gcs`) are sampled frequently (tens to
> hundreds of points); lab tests (`ag`…`scr`) a few times; blood counts
> (`plt`, `hb`, `wbc`) ~2 times; `weight` essentially once.

## Aggregating dynamic features to one row per patient
A model needs **one row per patient**, but each dynamic feature is a time series.
We must collapse each series to a fixed set of numbers. Options:

| Option | Pros | Cons |
|---|---|---|
| **First value** (the paper's choice) | Simple; the state at ICU admission | Discards most of the data; sensitive to a single reading |
| **Last value** | State closest to the prediction time | Still loses the trajectory |
| **Mean / median** | Stable, low-noise | **Hides the extremes**, which matter clinically |
| **{min, mean, max}** ✅ chosen | Keeps both the baseline and the extremes | More columns (×3); some redundancy |
| Full stats + slope/trend | Richest representation | Too many columns; overfitting risk on 970 rows |

**Chosen: `{min, mean, max}` per dynamic feature** (→ 17 × 3 = 51 dynamic columns).
Each timestamp is already within the first 24h of the ICU stay, so these stats
summarize that window.

**Why (evidence from this dataset):**
- In AKI, the **extremes carry the signal** — peak creatinine/BUN, and the
  *lowest* blood pressure (hypoperfusion precedes kidney injury).
- Concrete proof — correlation of `sbp` with the label by statistic:
  **min = −0.26, mean = +0.05, max = +0.27**. The mean almost entirely *hides*
  the relationship, while min and max each expose it (in opposite directions).
  A single first value or a mean-only summary would lose this.
- Vital signs have ~60 readings per patient in 24h, so reducing to one value
  wastes most of the data.
- It matches the paper's *spirit*: generate many candidate features, then let
  feature selection (LASSO) prune them — rather than pre-committing to one value.

**Trade-offs we accept:** more columns (handled by feature selection / tree
models), mild redundancy (e.g. `weight` has ~1 reading so its min=mean=max), and
mild multicollinearity between a feature's own min/mean/max (regularized and
tree-based models tolerate this). To reproduce the paper exactly, switch
`agg_series` to return the first value instead.

## Quick reference: counts
- Static features: **20** = 6 numeric scalars (`age`, `oasis`, `saps2`, `sofa`,
  `preiculos`, `egfr`) + 4 categoricals (`gender`, `race`, `dka_type`,
  `liver_disease`) + 3 always-present bools + 6 sparse comorbidity flags + `ckd_stage`.
- Dynamic features: **17**.
- Total distinct `measures` keys: **37** (identical key set in train and test).

*Clinical abbreviations cross-checked against the article's abbreviation list
(page 2). Items marked "inferred" should be confirmed against the original data
extraction if exact provenance matters.*
