## Danh Sách Kiểm Tra Trước Khi Nộp Bài

- [ ] **1. Định dạng file**: File nộp phải ở định dạng `.csv`.
- [ ] **2. Tên cột**: Đúng chính xác 2 cột: `id` và `revenue`.
- [ ] **3. Số lượng dòng**: Đúng chính xác số dòng của `test.csv` (3.869 dòng test hoặc tương ứng).
- [ ] **4. Giá trị Null/NaN**: Đảm bảo không có bất kỳ giá trị `NaN` hoặc `Null` nào trong cột `revenue`.
- [ ] **5. Giá trị âm**: Kiểm tra không có doanh thu dự đoán nào bị âm (`revenue >= 0`).
- [ ] **6. Biến đổi Inverse Log**: Đảm bảo đã dùng `np.expm1()` biến đổi ngược dự đoán từ mô hình học trên `log1p(revenue)`.
- [ ] **7. Tính tái lập**: Đã cố định `SEED = 42` trên toàn bộ mã nguồn.


## Danh Sách Kiểm Tra Trước Khi Nộp Bài

- [ ] **1. Định dạng file**: File nộp phải ở định dạng `.csv`.
- [ ] **2. Tên cột**: Đúng chính xác 2 cột: `customer_id` và `churn_3m`.
- [ ] **3. Số lượng dòng**: Đúng chính xác **8,000 dòng** tương ứng với `test.csv` (cộng thêm 1 dòng header).
- [ ] **4. Giá trị Null/NaN**: Đảm bảo không có bất kỳ giá trị `NaN` hoặc `Null` nào trong cột `churn_3m`.
- [ ] **5. Miền giá trị Xác suất**: Kiểm tra toàn bộ giá trị dự đoán `churn_3m` phải nằm trong khoảng $[0, 1]$ ($0.0 \le p \le 1.0$).
- [ ] **6. Thứ tự ID**: Kiểm tra danh sách `customer_id` trùng khớp hoàn toàn với thứ tự trong `test.csv`.
- [ ] **7. Tính tái lập**: Đã cố định `SEED = 42` trên toàn bộ mã nguồn huấn luyện.