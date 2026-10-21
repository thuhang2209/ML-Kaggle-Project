# Regression

## 1. So Sánh Hiệu Năng Các Mô Hình Trọng Điểm

| Mô hình | Val RMSLE | Ưu điểm | Nhược điểm |
| :--- | :---: | :--- | :--- |
| **Weekday Median** | 0.2387 | Đơn giản, không thể overfit | Không bắt được xu hướng & sự kiện |
| **LightGBM (Advanced Features)** | 0.1745 | Tốc độ nhanh, xử lý phi tuyến tốt | Nhạy cảm với tham số |
| **CatBoost (Advanced Features)** | 0.1698 | Xử lý tốt thuộc tính cửa hàng/thành phố | Huấn luyện lâu hơn LightGBM |
| **Weighted Ensemble (LGBM + CatBoost)** | **0.1632** | Giảm thiểu phương sai, độ ổn định cao | Tăng độ phức tạp của Pipeline |

## 2. Quyết Định Chọn Mô Hình Cuối Cùng
- **Lựa chọn chính**: **Weighted Ensemble (40% LightGBM + 60% CatBoost)**.
- **Lý do**:
  1. Điểm Validation RMSLE đạt mức thấp nhất ($0.1632$).
  2. Hai mô hình có tương quan dự đoán dưới $0.92$, hỗ trợ và bù trừ sai số cho nhau tốt.
  3. Mô hình CatBoost nắm bắt tốt thông tin cửa hàng, trong khi LightGBM tối ưu tốt các lag feature thời gian.

# Classification

## 1. So Sánh Hiệu Năng Các Mô Hình Trọng Điểm

| Mô hình | Val ROC-AUC (5-Fold) | Ưu điểm | Nhược điểm |
| :--- | :---: | :--- | :--- |
| **Logistic Regression** | 0.7290 | Đơn giản, tính toán nhanh, dễ giải thích | Không bắt được mối quan hệ phi tuyến[cite: 8, 9] |
| **LightGBM (Engineered)** | 0.8515 | Tốc độ nhanh, xử lý tốt dữ liệu khuyết thiếu | Nhạy cảm với nhiễu nếu depth quá cao |
| **CatBoost (Engineered)** | 0.8560 | Tối ưu tuyệt vời cho biến danh mục (`province`, `occupation`) | Thời gian huấn luyện lâu hơn LightGBM |
| **Rank Ensemble (LGBM + CatBoost)** | **0.8625** | Tối đa hóa ROC-AUC, giảm phương sai dự đoán | Tăng độ phức tạp của Pipeline |

## 2. Quyết Định Chọn Mô Hình Cuối Cùng
- **Lựa chọn chính**: **Rank Ensemble (40% LightGBM + 60% CatBoost)**.
- **Lý do**:
  1. Điểm Validation 5-Fold ROC-AUC đạt mức cao nhất ($0.8625$).
  2. Rank Averaging giúp kết hợp thứ tự xếp hạng xác suất của 2 mô hình một cách mượt mà mà không lo chênh lệch thang đo xác suất.