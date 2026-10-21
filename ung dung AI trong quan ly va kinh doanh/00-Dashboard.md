# Dashboard Quản Lý Dự Án ML: Dự Đoán Doanh Thu Chuỗi Cửa Hàng Cafe

## 1. Tổng Quan Dự Án & Mục Tiêu
- **Bài toán**: Forecasting / Regression — Dự đoán doanh thu hàng ngày (`revenue`, triệu VNĐ) cho từng cửa hàng cafe từ **01/01/2026** đến **31/03/2026**.
- **Metric chính thức**: **RMSLE** (Root Mean Squared Error trên `log(1 + revenue)`). **Metric càng thấp càng tốt**.
- **Mục tiêu Kaggle**: Xây dựng pipeline xử lý dữ liệu, tối ưu feature engineering, cross-validation chuẩn mực và ensemble mô hình để đạt điểm RMSLE tối ưu.

## 2. Liên Kết Nhanh Trong Vault
- [[01-Phan-cong-5-thanh-vien|Phân công nhiệm vụ 5 thành viên]]
- [[02-Ke-hoach-thuc-hien|Kế hoạch thực hiện & Timeline]]
- [[03-Experiment-Tracker|Bảng theo dõi thử nghiệm (Experiment Tracker)]]
- [[04-Ket-qua-va-quyet-dinh|Kết quả & Quyết định chọn mô hình]]
- [[05-Kaggle-Submission-Checklist|Checklist kiểm tra file Submission]]

## 3. Trạng Thái Các Luồng Công Việc
- [ ] **EDA & Data Quality**: Khám phá cấu trúc 5 tập dữ liệu, phát hiện nhiễu/outliers và định dạng mở rộng.
- [ ] **Validation Strategy**: Thiết lập Time-based Validation (T10/2025 - T12/2025).
- [ ] **Baseline Pipeline**: Xây dựng mô hình Baseline (Median/LightGBM cơ bản).
- [ ] **Boosting & Tuning**: Thử nghiệm LightGBM, XGBoost, CatBoost.
- [ ] **Feature Engineering**: Tạo lag features, rolling statistics, holiday distance, weather impact.
- [ ] **Ensembling**: Thử nghiệm Weighted Averaging / Stacking.
- [ ] **Submission Check**: Verify ID, row count, non-negative prediction.

## Dashboard Quản Lý Dự Án ML: Dự Đoán Rời Bỏ Thẻ Tín Dụng (Card Churn)

## 1. Tổng Quan Dự Án & Mục Tiêu
- **Bài toán**: Binary Classification — Dự đoán xác suất khách hàng hủy thẻ tín dụng trong 3 tháng tới (`churn_3m`).
- **Metric chính thức**: **ROC-AUC**. **Metric CÀNG CAO CÀNG TỐT** (0.5 = ngẫu nhiên, 1.0 = hoàn hảo).
- **Mục tiêu Kaggle**: Xây dựng pipeline làm sạch dữ liệu nhiễu, tạo đặc trưng tài chính/hành vi, Stratified 5-Fold CV và Ensemble mô hình Boosting để tối ưu ROC-AUC.

## 2. Liên Kết Nhanh Trong Vault
- [[01-Phan-cong-5-thanh-vien|Phân công nhiệm vụ 5 thành viên]]
- [[02-Ke-hoach-thuc-hien|Kế hoạch thực hiện & Timeline]]
- [[03-Experiment-Tracker|Bảng theo dõi thử nghiệm Classification]]
- [[04-Ket-qua-va-quyet-dinh|Kết quả & Quyết định chọn mô hình]]
- [[05-Kaggle-Submission-Checklist|Checklist kiểm tra file Submission]]

## 3. Trạng Thái Các Luồng Công Việc
- [ ] **EDA & Data Quality**: Phát hiện nhiễu chữ (gender, province), tỷ lệ churn 11.7% và dữ liệu khuyết thiếu.
- [ ] **Validation Strategy**: Thiết lập Stratified 5-Fold Cross-Validation.
- [ ] **Baseline Pipeline**: Xây dựng Logistic Regression Baseline (5-fold AUC ~0.729).
- [ ] **Tree / Boosting Models**: Huấn luyện Random Forest, LightGBM, CatBoost.
- [ ] **Feature Engineering**: Tạo feature chỉ số tài chính, tần suất tương tác app, lịch sử khiếu nại.
- [ ] **Probability Ensembling**: Rank Averaging / Blending xác suất từ các mô hình đơn.
- [ ] **Submission Check**: Đảm bảo ID, 8,000 dòng test, xác suất p nằm trong [0, 1].

---
*Ghi chú*: Cập nhật tiến độ hàng ngày tại [[03-Experiment-Tracker]].
---
