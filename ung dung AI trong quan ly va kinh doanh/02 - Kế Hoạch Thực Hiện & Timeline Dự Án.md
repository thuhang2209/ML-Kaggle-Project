
# Bai toan Regression
## 1. Các Giai Đoạn Thực Hiện

### Giai đoạn 1: Chuẩn bị & Khám phá dữ liệu (Ngày 1 - 2)
- [ ] Khám phá dữ liệu (EDA), kiểm tra kiểu dữ liệu, missing values.
- [ ] Thiết lập chiến lược Validation trùng khớp với bài toán Time-Series.
- [ ] Cố định ngẫu nhiên `SEED = 42` cho toàn hệ thống.

### Giai đoạn 2: Baseline & Tiền xử lý dữ liệu (Ngày 3 - 4)
- [ ] Làm sạch dữ liệu: Chuẩn hóa ngày tháng, fill missing values thời tiết.
- [ ] Xây dựng mô hình Baseline (Median) & ML cơ bản.
- [ ] Tạo file nộp bài đầu tiên (`baseline_submission.csv`).

### Giai đoạn 3: Feature Engineering & Boosting (Ngày 5 - 8)
- [ ] Xây dựng Lag & Rolling Features.
- [ ] Xây dựng Weather & Holiday features (đặc biệt là Tết Âm Lịch 2026).
- [ ] Thử nghiệm & Tuning LightGBM, CatBoost, XGBoost.

### Giai đoạn 4: Ensemble & Kiểm tra mô hình (Ngày 9 - 10)
- [ ] So sánh tương quan dự đoán giữa các mô hình.
- [ ] Thử nghiệm Weighted Averaging / Stacking.
- [ ] Kiểm tra xem Ensemble có thực sự cải thiện CV hay không.

### Giai đoạn 5: Submission & Đóng gói (Ngày 11)
- [ ] Chạy checklist kiểm tra file submission.
- [ ] Nộp kết quả lên Kaggle và đối chiếu với Public Leaderboard.

## 2. Rủi Ro & Biện Pháp Giảm Thao Tác
- **Data Leakage**: Sử dụng dữ liệu tương lai để dự đoán quá khứ $\rightarrow$ *Giải pháp*: Chỉ dùng thông tin tính đến thời điểm $t-1$ cho lag/rolling features.
- **Chệch do Tết Nguyên Đán**: Tết 2026 rơi vào Quý 1 (thời gian test) $\rightarrow$ *Giải pháp*: Feature khoảng cách tới Tết Âm Lịch thay vì ngày dương lịch cố định.
- **Overfitting trên Public Leaderboard**: *Giải pháp*: Tin tưởng hoàn toàn vào kết quả Validation cục bộ (Local CV).

# Bai toan Classification

# 02 — Kế Hoạch Thực Hiện & Timeline Bài Toán Classification

## 1. Các Giai Đoạn Thực Hiện

### Giai đoạn 1: Khám phá & Làm sạch dữ liệu (Ngày 1 - 2)
- [ ] Phân tích tỷ lệ mất cân bằng lớp (Churn Rate = 11.7%).
- [ ] Chuẩn hóa dữ liệu chữ (`gender`, `province`, `occupation`).
- [ ] Thiết lập Stratified 5-Fold CV và cố định `SEED = 42`.

### Giai đoạn 2: Baseline & Mô hình ML truyền thống (Ngày 3 - 4)
- [ ] Chạy pipeline Baseline Logistic Regression (Target: AUC 0.729).
- [ ] Thử nghiệm Random Forest & Decision Tree.
- [ ] Tạo file nộp bài đầu tiên (`baseline_sub.csv`).

### Giai đoạn 3: Feature Engineering & Boosting (Ngày 5 - 8)
- [ ] Tạo các đặc trưng về tỷ lệ chi tiêu, khiếu nại, hạn mức thẻ.
- [ ] Huấn luyện & Tối ưu LightGBM, CatBoost, XGBoost với Optuna.
- [ ] Đánh giá điểm ROC-AUC trên 5 Folds.

### Giai đoạn 4: Ensembling & Blending Xác Suất (Ngày 9 - 10)
- [ ] Tính tương quan Rank Correlation giữa dự đoán của các mô hình.
- [ ] Thực hiện Rank Averaging / Weighted Blending xác suất.
- [ ] Kiểm tra điểm AUC CV để đảm bảo không bị Overfitting.

### Giai đoạn 5: Kiểm tra & Nộp bài Kaggle (Ngày 11)
- [ ] Chạy checklist kiểm tra file submission (8,000 dòng, $p \in [0, 1]$).
- [ ] Nộp bài lên Kaggle và đối chiếu với Public Leaderboard.

## 2. Quản Trị Rủi Ro Bài Toán Classification
- **Data Leakage trong Preprocessing**: Impute missing values/Scaling bằng dữ liệu của cả tập test $\rightarrow$ *Giải pháp*: Chỉ fit Scaler/Imputer trên tập Train của từng Fold.
- **Mất cân bằng lớp khiến mô hình dự đoán $p \approx 0$**: $\rightarrow$ *Giải pháp*: Tinh chỉnh `scale_pos_weight` hoặc đánh giá mô hình bằng ROC-AUC (vốn không nhạy cảm với chọn ngưỡng nhị phân).