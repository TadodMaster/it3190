# Nhật ký tinh chỉnh Random Forest (DKA-AKI)

Dữ liệu: 970 × 77 | dương tính 38.9% | CV: Stratified 5-fold | độ đo chọn mô hình: **ROC-AUC** | `random_state=42`.

Sinh tự động bởi notebook `modeling_randomforest.ipynb` mục 7.

---

## Baseline (cấu hình hiện tại — mục 4)  
_2026-06-16 15:08:04_
- **Thay đổi so với baseline:** (mốc tham chiếu)
- **Tham số:** n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, bootstrap=True, class_weight=balanced
- **CV ROC-AUC:** 0.8042 ± 0.0438
- **Validation @ngưỡng 0.50:** AUC=0.806, Accuracy=0.753, Sensitivity=0.667, Specificity=0.807, F1=0.676
- **Ghi chú:** Cấu hình đang dùng trong notebook trước khi tinh chỉnh.

## GridSearch — cấu hình tốt nhất  
_2026-06-16 15:53:03_
- **Thay đổi so với baseline:** n_estimators: 300 → 500; max_features: sqrt → log2
- **Tham số:** n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features=log2, bootstrap=True, class_weight=balanced
- **CV ROC-AUC:** 0.8109 ± 0.0445
- **Validation @ngưỡng 0.50:** AUC=0.807, Accuracy=0.758, Sensitivity=0.667, Specificity=0.815, F1=0.680
- **Ghi chú:** Tốt nhất trong 1080 tổ hợp theo CV ROC-AUC.

## Best RF + ngưỡng tối đa F1 (0.470)  
_2026-06-16 15:53:03_
- **Thay đổi so với baseline:** ngưỡng 0.5 → 0.470
- **Tham số:** n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features=log2, bootstrap=True, class_weight=balanced
- **CV ROC-AUC:** 0.8109 ± 0.0445
- **Validation @ngưỡng 0.47:** AUC=0.807, Accuracy=0.763, Sensitivity=0.707, Specificity=0.798, F1=0.697
- **Ghi chú:** Cùng mô hình tốt nhất, chỉ thay đổi ngưỡng quyết định.

## Best RF + ngưỡng Sens>=0.85 (0.330)  
_2026-06-16 15:53:03_
- **Thay đổi so với baseline:** ngưỡng 0.5 → 0.330
- **Tham số:** n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features=log2, bootstrap=True, class_weight=balanced
- **CV ROC-AUC:** 0.8109 ± 0.0445
- **Validation @ngưỡng 0.33:** AUC=0.807, Accuracy=0.670, Sensitivity=0.853, Specificity=0.555, F1=0.667
- **Ghi chú:** Cùng mô hình tốt nhất, chỉ thay đổi ngưỡng quyết định.

