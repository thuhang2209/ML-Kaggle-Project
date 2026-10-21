## Bai toan: Regression
### Thành Viên 1 — EDA & Chất Lượng Dữ Liệu
- **Mục tiêu**: Nắm trọn vẹn đặc trưng dữ liệu, phát hiện dữ liệu lỗi/nhiễu/outlier, phòng tránh leakage và đề xuất ý tưởng Feature Engineering.
- **Đầu việc chi tiết**:
  1. Phân tích phân phối biến mục tiêu `revenue` (phát hiện giá trị cực đại $53.600$ triệu VNĐ).
  2. Kiểm tra dữ liệu thiếu, định dạng ngày tháng (`open_date` hỗn hợp YYYY-MM-DD và DD/MM/YYYY).
  3. Phân tích tác động của thời tiết (`weather.csv`), ngày lễ/Tết Nguyên Đán 2026 (`holidays.csv`).
- **File bàn giao**: `eda_report.ipynb`, `data_cleaning_rules.py`.
- **Tiêu chí hoàn thành**: Báo cáo EDA đầy đủ biểu đồ, sạch dữ liệu đầu vào.
- **Phụ thuộc**: Không.
- **Mức ưu tiên**: High.

### Thành Viên 2 — Mô Hình Baseline & ML Truyền Thống
- **Mục tiêu**: Xây dựng pipeline baseline hoàn chỉnh, tạo mốc điểm RMSLE ban đầu để cả nhóm so sánh công bằng.
- **Đầu việc chi tiết**:
  1. Thiết lập mô hình Baseline ngô nghê (Median theo `store_id` & `weekday`).
  2. Thử nghiệm các mô hình ML truyền thống: Linear Regression, Ridge, Lasso, DecisionTree.
  3. Đảm bảo toàn bộ mô hình huấn luyện trên nhãn $\log(1 + y)$ và biến đổi ngược $\exp(p) - 1$.
- **File bàn giao**: `baseline_pipeline.py`, `baseline_submission.csv`.
- **Tiêu chí hoàn thành**: Pipeline chạy thông suốt không lỗi, nộp thành công file baseline lên Kaggle.
- **Phụ thuộc**: Thành viên 1.
- **Mức ưu tiên**: High.

### Thành Viên 3 — Boosting & Tuning
- **Mục tiêu**: Huấn luyện và tinh chỉnh các mô hình Gradient Boosting hàng đầu (LightGBM, XGBoost, CatBoost).
- **Đầu việc chi tiết**:
  1. Xây dựng mô hình LightGBM, XGBoost, CatBoost trên bộ feature chuẩn.
  2. Tối ưu Hyperparameters bằng Optuna/GridSearch (learning_rate, max_depth, num_leaves, subsample).
  3. Xử lý categorical features trực tiếp bằng CatBoost/LightGBM native handling.
- **File bàn giao**: `model_lightgbm.py`, `model_xgboost.py`, `model_catboost.py`.
- **Tiêu chí hoàn thành**: Đạt RMSLE CV thấp hơn mô hình Baseline rõ rệt.
- **Phụ thuộc**: Thành viên 2, Thành viên 5.
- **Mức ưu tiên**: High.

### Thành Viên 4 — Feature Engineering & Ensemble
- **Mục tiêu**: Tạo các đặc trưng (features) có giá trị cao và thử nghiệm các phương pháp Ensemble để tối ưu điểm số.
- **Đầu việc chi tiết**:
  1. Tạo Time-series Features: Lag 7, 14, 28, 364 ngày; Rolling Mean/Std 7, 14, 30 ngày.
  2. Tạo Event Features: Khoảng cách tới Tết Âm Lịch 2026, ngày lễ quốc gia.
  3. Kết hợp mô hình (Weighted Average, Blending, Stacking) dựa trên mức độ bổ sung lẫn nhau.
- **File bàn giao**: `feature_factory.py`, `ensemble_pipeline.py`.
- **Tiêu chí hoàn thành**: Thêm feature giúp CV giảm; Ensemble có CV tốt hơn bất kỳ single model nào.
- **Phụ thuộc**: Thành viên 1, Thành viên 3.
- **Mức ưu tiên**: Medium-High.

### Thành Viên 5 — Validation, Tracking & Submission
- **Mục tiêu**: Thiết lập chiến lược Validation nghiêm ngặt, quản lý bảng kết quả, kiểm tra tính tái lập (reproducibility) và đảm bảo định dạng nộp bài.
- **Đầu việc chi tiết**:
  1. Thiết lập Time-based Out-of-Time Validation (Dùng 01/10/2025 - 31/12/2025 làm Val Set).
  2. Cố định `random_state`/`seed` cho toàn đội.
  3. Quản lý [[03-Experiment-Tracker]] và kiểm tra định dạng `submission.csv` (`id`, `revenue`).
- **File bàn giao**: `validation_split.py`, [[03-Experiment-Tracker]], `submission_checker.py`.
- **Tiêu chí hoàn thành**: Không bị Data Leakage; mọi thử nghiệm đều được ghi log minh bạch.
- **Phụ thuộc**: Tham gia xuyên suốt từ đầu.
- **Mức ưu tiên**: High.

## Bài Toán Classification

### Thành Viên 1 — EDA & Chất Lượng Dữ Liệu
- **Mục tiêu**: Chuẩn hóa dữ liệu thô/nhiễu, xử lý ô khuyết thiếu (missing values) và đưa ra insights về hành vi churner.
- **Đầu việc chi tiết**:
  1. Chuẩn hóa chuỗi văn bản: `gender` ('F', 'female', 'Nữ' -> 'F'), `province` ('TP.HCM', 'TP HCM' -> 'HCM').
  2. Phân tích phân phối ô trống (`other_bank_cards` thiếu 20.3%, `income_month` thiếu 12.9%).
  3. Xử lý giá trị bất thường (outliers): Tuổi = 999, Thu nhập/Thẻ âm.
- **File bàn giao**: `eda_churn.ipynb`, `clean_data.py`.
- **Tiêu chí hoàn thành**: Bảng dữ liệu sạch, không bị mất thông tin quan trọng.
- **Phụ thuộc**: Không. | **Ưu tiên**: High.

### Thành Viên 2 — Mô Hình Baseline & ML Truyền Thống
- **Mục tiêu**: Xây dựng pipeline Baseline và các mô hình ML truyền thống với Stratified 5-Fold CV.
- **Đầu việc chi tiết**:
  1. Tái lập mô hình Baseline Logistic Regression (Mean AUC 0.729).
  2. Thử nghiệm Random Forest, Decision Tree, Naive Bayes trên tập đặc trưng số.
  3. Đảm bảo mô hình xuất ra xác suất $P(\text{churn} = 1)$ liên tục.
- **File bàn giao**: `baseline_classification.py`, `baseline_churn_sub.csv`.
- **Tiêu chí hoàn thành**: Pipeline chạy ổn định, nộp bài thành công trên Kaggle.
- **Phụ thuộc**: Thành viên 1. | **Ưu tiên**: High.

### Thành Viên 3 — Boosting & Tuning
- **Mục tiêu**: Huấn luyện và tinh chỉnh các mô hình Gradient Boosting mạnh nhất cho dữ liệu bảng (LightGBM, XGBoost, CatBoost).
- **Đầu việc chi tiết**:
  1. Xây dựng LightGBM, XGBoost, CatBoost tích hợp Stratified 5-Fold CV.
  2. Xử lý mất cân bằng lớp bằng tham số `scale_pos_weight` hoặc `is_unbalance`.
  3. Dùng Optuna tối ưu hyperparameters (`max_depth`, `learning_rate`, `subsample`).
- **File bàn giao**: `model_lgbm.py`, `model_catboost.py`, `model_xgb.py`.
- **Tiêu chí hoàn thành**: Đạt AUC CV cao hơn mô hình Baseline rõ rệt (> 0.82).
- **Phụ thuộc**: Thành viên 2, Thành viên 5. | **Ưu tiên**: High.

### Thành Viên 4 — Feature Engineering & Ensemble
- **Mục tiêu**: Tạo ra các đặc trưng có giá trị dự báo cao và kết hợp xác suất (Probability Ensembling).
- **Đầu việc chi tiết**:
  1. Tạo các chỉ số tài chính: Tỷ lệ sử dụng hạn mức (`utilization_6m`), Biến động chi tiêu (`spend_change_pct`).
  2. Tạo chỉ số tương tác: Tỷ lệ đăng nhập App (`app_logins_30d`), Khiếu nại & Trả chậm (`complaints_12m` + `late_payments_12m`).
  3. Thực hiện Rank Averaging / Blending xác suất giữa LightGBM và CatBoost.
- **File bàn giao**: `feature_engineering.py`, `ensemble_churn.py`.
- **Tiêu chí hoàn thành**: Ensemble giúp tăng AUC CV so với mô hình đơn tốt nhất.
- **Phụ thuộc**: Thành viên 1, Thành viên 3. | **Ưu tiên**: Medium-High.

# Thành Viên 5 — Validation, Tracking & Submission
- **Mục tiêu**: Đảm bảo quy trình Stratified 5-Fold CV nghiêm ngặt, chống Data Leakage và quản lý bảng Experiment Tracker.
- **Đầu việc chi tiết**:
  1. Thiết lập cố định `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
  2. Quản lý [[03-Experiment-Tracker]] (Ghi log ROC-AUC CV).
  3. Kiểm tra định dạng `submission.csv` (8,000 dòng test, cột `customer_id` và `churn_3m`).
- **File bàn giao**: `cv_checker.py`, [[03-Experiment-Tracker]], `sub_verifier.py`.
- **Tiêu chí hoàn thành**: Kiểm soát chất lượng dự đoán, đảm bảo tính tái lập (SEED = 42).
- **Phụ thuộc**: Tham gia xuyên suốt từ đầu. | **Ưu tiên**: High.