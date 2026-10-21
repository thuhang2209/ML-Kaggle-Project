# Classification Project Plan — Credit Card Churn Prediction

> **Vai trò:** Senior Data Scientist  
> **Bài toán:** Binary classification — dự đoán khách hàng có đóng thẻ tín dụng trong 3 tháng tiếp theo hay không.  
> **Dataset chính:** `train.csv`, `sample_submission.csv`, `starter_churn.ipynb`.  
> **Ngày snapshot theo notebook:** 30/06/2026.  
> **Metric chính theo notebook:** ROC AUC.

---

## 1. Executive Summary

Xây dựng mô hình phân loại nhị phân ước lượng xác suất khách hàng đóng thẻ tín dụng trong 3 tháng sau ngày snapshot. Mô hình phải xếp hạng khách hàng có nguy cơ churn cao lên trên khách hàng có nguy cơ thấp, có khả năng tổng quát hóa trên khách hàng chưa từng xuất hiện trong train, và tạo được file submission đúng định dạng yêu cầu.

### 1.1 Định nghĩa bài toán

- **Observation unit:** một khách hàng tại thời điểm snapshot.
- **Target:** `churn_3m`.
  - `1`: khách hàng đóng thẻ trong 3 tháng sau snapshot.
  - `0`: khách hàng không đóng thẻ trong khoảng thời gian đó.
- **Output của mô hình:** xác suất `P(churn_3m = 1 | customer features)` trong khoảng `[0, 1]`, không chỉ là nhãn 0/1.
- **Metric chính:** ROC AUC, theo mô tả trong `starter_churn.ipynb`.
- **Use case kinh doanh giả định cần xác nhận:** xếp hạng danh sách khách hàng để ưu tiên chăm sóc/retention. Chưa có thông tin về chi phí giữ chân, năng lực chiến dịch hoặc ngưỡng can thiệp; không nên tự chọn threshold kinh doanh trước khi có các thông tin đó.

### 1.2 Những gì đã quan sát trực tiếp trong file

| Hạng mục | Kết quả kiểm tra ban đầu |
|---|---|
| Train | 12.120 dòng, 24 cột gồm `customer_id`, các feature và `churn_3m` |
| Customer ID | 12.000 ID khác nhau; có 120 dòng lặp ID và 120 dòng trùng lặp hoàn toàn |
| Target | `churn_3m` có hai nhãn 0/1; trên file thô có 10.707 nhãn 0, 1.413 nhãn 1, tỷ lệ dương khoảng 11,66% |
| Missing values | Nhiều feature thiếu dữ liệu; nhiều nhất là `other_bank_cards` (20,32%), `income_month` (12,89%), `app_logins_30d` (12,51%) |
| Text/categorical | Một số biến có nhiều cách viết hoa/thường, dấu cách và bí danh địa danh |
| Date | `open_date` chứa nhiều định dạng khác nhau: `YYYY-MM-DD`, `YYYYMMDD`, `DD/MM/YYYY` |
| Giá trị cần điều tra | `age` tối đa 999; `income_month` và `credit_limit` có cực trị lớn; `utilization_6m` có giá trị âm và lớn hơn 1; `txn_amount_3m` có giá trị âm |
| `bank_code` | Chỉ có một giá trị quan sát được: `VB01` |
| Submission template | 8.000 dòng, hai cột `customer_id`, `churn_3m`; cột `churn_3m` hiện có giá trị mẫu cố định `0.12` |
| Test set | Chưa thấy file `test.csv` trong những file được cung cấp ở lượt này; notebook mẫu có tham chiếu đến `test.csv` |

**Lưu ý:** số liệu trên phản ánh file train thô trước khi xử lý. Mọi thống kê, tỷ lệ target và phân phối feature phải được tính lại sau khi xử lý duplicate và xác nhận quy tắc làm sạch.

### 1.3. Chia dữ liệu thành train / validation / test

Đã tạo split nội bộ từ `train(1).csv` sau khi loại **120 dòng trùng lặp hoàn toàn**. Vì dataset là dữ liệu khách hàng dạng cross-sectional với một snapshot cho mỗi khách hàng sau dedup và target có mất cân bằng, dùng **stratified random split 70/15/15** để giữ tỷ lệ churn gần tương đương giữa các tập. Seed cố định là `42` để kết quả tái lập được.

| Tập | Số dòng | Tỷ lệ | Churn = 1 | Churn rate |
|---|---:|---:|---:|---:|
| Train | 8.400 | 70% | 984 | 11,714% |
| Validation | 1.800 | 15% | 211 | 11,722% |
| Test nội bộ | 1.800 | 15% | 210 | 11,667% |
| **Tổng sau dedup** | **12.000** | **100%** | **1.405** | **11,708%** |

Số lớp còn lại: train có 7.416 nhãn 0; validation có 1.589 nhãn 0; test có 1.590 nhãn 0. Sau exact dedup, `customer_id` duy nhất trên 12.000 dòng. Nếu còn ID trùng nhưng dữ liệu không giống hệt nhau ở một phiên bản dataset khác, dừng split và xác minh snapshot/record trước khi tiếp tục.

Các file đã sinh trong thư mục `classification_splits/`: `train.csv`, `val.csv`, `test.csv`. File gốc `train(1).csv` không bị ghi đè. Mã tái tạo split:

```python
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

source = pd.read_csv("train(1).csv").drop_duplicates()
if source["customer_id"].duplicated().any():
    raise ValueError("Một customer_id có nhiều record khác nhau; cần xác minh trước khi split")
if source["churn_3m"].isna().any() or not set(source["churn_3m"].unique()).issubset({0, 1}):
    raise ValueError("Target churn_3m không hợp lệ")

train, rest = train_test_split(
    source, test_size=0.30, stratify=source["churn_3m"], random_state=42
)
val, test = train_test_split(
    rest, test_size=0.50, stratify=rest["churn_3m"], random_state=42
)
out = Path("classification_splits")
out.mkdir(parents=True, exist_ok=True)
for name, part in {"train": train, "val": val, "test": test}.items():
    part.to_csv(out / f"{name}.csv", index=False)
```

**Cách dùng tập:** fit preprocessing và mô hình chỉ trong train; dùng Stratified 5-fold CV bên trong train để chọn ứng viên/hyperparameter; dùng validation để so sánh cuối, chọn calibration/threshold khi có yêu cầu nghiệp vụ; sau khi khóa toàn bộ quyết định mới chấm test nội bộ đúng một lần. Test nội bộ vẫn có label vì nó được lấy từ file train đã gán nhãn, nên khác với `test.csv` benchmark/Kaggle (chưa được cung cấp) vốn không có target. Không dùng test nội bộ để tuning.

---

## 2. Phạm vi, mục tiêu và tiêu chí thành công

### 2.1 Phạm vi triển khai

1. Audit cấu trúc và chất lượng dữ liệu.
2. Làm sạch dữ liệu nhưng giữ lại dấu vết để tái lập.
3. Thiết kế pipeline tiền xử lý không rò rỉ dữ liệu.
4. Xây dựng baseline dễ giải thích.
5. So sánh mô hình tuyến tính và mô hình cây/boosting.
6. Dùng 70% dữ liệu làm train để fit và 5-fold Stratified CV bên trong train; dùng validation 15% để chọn mô hình, còn test 15% được giữ kín cho đánh giá cuối cùng bằng ROC AUC.
7. Giải thích mô hình, phân tích lỗi và xác định threshold nếu có mục tiêu kinh doanh.
8. Huấn luyện mô hình cuối cùng, dự đoán tập test và xuất submission.
9. Thiết lập kiểm thử dữ liệu, khả năng tái lập và giám sát drift sau triển khai.

### 2.2 Tiêu chí thành công kỹ thuật

- Có thể tái tạo kết quả từ cùng dữ liệu, cấu hình và seed.
- ROC AUC được báo cáo qua 5 fold stratified bên trong tập train 70% (mean và độ lệch chuẩn), sau đó báo cáo validation riêng; test holdout 15% chỉ đánh giá một lần sau khi khóa mô hình.
- Tiền xử lý được fit bên trong từng fold, tránh dùng thống kê từ validation fold.
- Không có target leakage; mọi feature phải có sẵn tại thời điểm snapshot.
- Xử lý duplicate có quy tắc rõ ràng và không nhân đôi trọng số khách hàng.
- Output có đúng 8.000 ID của test nếu kích thước này được xác nhận khi nhận `test.csv`, đúng thứ tự theo template, không thiếu/nhân đôi ID, không có xác suất ngoài `[0, 1]` hoặc giá trị thiếu.
- Có báo cáo mô hình, feature importance/giải thích, giới hạn sử dụng và checklist nghiệm thu.

**Không đặt mục tiêu ROC AUC cụ thể trước khi chạy baseline**. Notebook được cung cấp chỉ mô tả cách đo baseline, không chứa một kết quả AUC đã được xác minh trong các output mà ta kiểm tra được.

---

## 3. Hiểu dữ liệu và data dictionary

Các feature hiện có trong `train(1).csv`:

| Cột | Nhóm | Hướng xử lý ban đầu |
|---|---|---|
| `customer_id` | ID | Giữ lại để liên kết output; loại khỏi feature set ban đầu vì là định danh, có nguy cơ học thuộc ID |
| `bank_code` | Categorical | Chỉ thấy `VB01`; nếu thực sự hằng số thì loại khỏi mô hình |
| `age` | Numeric | Xác minh giá trị phi thực tế/sentinel, missing và outlier |
| `gender` | Categorical | Chuẩn hóa alias/hoa thường/khoảng trắng; missing riêng |
| `province` | Categorical | Chuẩn hóa bí danh địa danh, dấu và cách viết; không gộp nếu chưa chắc cùng địa phương |
| `occupation` | Categorical | Chuẩn hóa các nhãn như `salaried`, `Salaried`, `salaried `; phân biệt `other`, `retired`, `student` |
| `income_month` | Numeric | Điều tra đơn vị, missing và cực trị; cân nhắc log transform sau xác minh |
| `card_type` | Categorical | Giữ làm feature; kiểm tra nhãn và tỷ lệ mỗi lớp |
| `open_date` | Date/string | Parse bằng định dạng tường minh rồi tạo tenure tại snapshot; audit ngày không hợp lệ |
| `credit_limit` | Numeric | Điều tra khác biệt đơn vị và cực trị trước khi winsorize/cap |
| `annual_fee_waived` | Binary | Xác minh chỉ có 0/1; giữ dạng binary |
| `utilization_6m` | Numeric | Xác minh miền giá trị, đặc biệt giá trị âm và trên 1 |
| `txn_count_3m` | Count | Kiểm tra giá trị âm/không nguyên/ngoại lệ |
| `txn_amount_3m` | Numeric | Xác minh ý nghĩa của số âm (ví dụ hoàn tiền/điều chỉnh) trước khi sửa hoặc loại |
| `spend_change_pct` | Numeric | Kiểm tra phạm vi và cách mã hóa phần trăm |
| `app_logins_30d` | Count | Impute và kiểm tra giá trị bất thường |
| `complaints_12m` | Count | Giữ dạng count; kiểm tra phạm vi |
| `late_payments_12m` | Count | Giữ dạng count; kiểm tra phạm vi |
| `salary_via_bank` | Categorical/binary | Chuẩn hóa `Y`/`N`; missing thành nhóm riêng hoặc impute phù hợp |
| `other_bank_cards` | Count | Xem xét là count, missing indicator và imputation |
| `promo_ended_60d` | Binary | Giữ làm feature nếu xác nhận đã biết tại snapshot |
| `last_contact_channel` | Categorical | Giữ các kênh `app`, `branch`, `hotline`, `none`; phân biệt `none` với missing |
| `currency` | Categorical | Chuẩn hóa thành chữ hoa; nếu chỉ còn một mức thì loại khỏi mô hình |
| `churn_3m` | Target | Chỉ dùng làm `y`, không được đưa vào `X` |

### 3.1 Target và class imbalance

Trên file thô, tỷ lệ lớp churn = 1 khoảng 11,66%. Đây là bài toán mất cân bằng lớp ở mức cần lưu ý; accuracy đơn thuần không phản ánh tốt chất lượng xếp hạng churn.

Việc đầu tiên sau dedup là tính lại:

- số lượng khách hàng ở mỗi nhãn;
- tỷ lệ churn;
- tỷ lệ churn theo các phân khúc lớn;
- số lượng target bị thiếu hoặc nhãn không thuộc `{0, 1}`.

Kiểm tra xem tỷ lệ churn giữa train và các fold có ổn định không. Không cân bằng dữ liệu bằng cách oversampling trước khi cross-validation, vì cách đó có thể làm bản sao của cùng quan sát lọt sang cả train và validation.

---

## 4. Phase 0 — Chuẩn bị môi trường và quản lý file

### Công việc

1. Tạo cấu trúc dự án, cố định phiên bản thư viện và random seed.
2. Giữ nguyên file gốc; tạo bản làm việc đã chuẩn hóa tên file.
3. Xác nhận `test.csv` từ nguồn dữ liệu trước khi xây dựng submission cuối cùng.
4. Lưu cấu hình, log, metric và model artifact ở các thư mục riêng.

Notebook `starter_churn.ipynb` đang tìm `train.csv` và `test.csv`, trong khi các file mới tải lên có tên `train(1).csv` và `sample_submission(1).csv`. Hãy copy/đổi tên file trong thư mục làm việc theo quy ước thay vì ghi đè file gốc hoặc vô tình dùng nhầm dataset khác.

### Cấu trúc thư mục đề xuất

```text
credit-card-churn/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── config.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── predict.py
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── submissions/
├── tests/
├── requirements.txt
├── README.md
└── Classification_Plan.md
```

### Deliverables

- Môi trường chạy được từ đầu đến cuối.
- Data manifest gồm tên file, số dòng/cột, thời điểm tạo và hash nếu cần.
- Script hoặc notebook EDA có thể chạy lại.

---

## 5. Phase 1 — Data validation, deduplication và quality audit

### 5.1 Kiểm tra schema

- Kiểm tra danh sách cột, dtype, số dòng, encoding và giá trị target.
- Xác nhận `customer_id` không null và kiểm tra tính duy nhất.
- Xác nhận `churn_3m` chỉ có `0` và `1`.
- So sánh schema train/test; test không được có target.
- Kiểm tra template có đúng hai cột `customer_id`, `churn_3m`.

### 5.2 Duplicate — xử lý trước khi chia fold

File thô có 120 dòng trùng lặp hoàn toàn; số ID khác nhau là 12.000 và các dòng lặp được kiểm tra có cùng dữ liệu/target. Quy trình đề xuất:

1. Ghi lại số dòng trước xử lý.
2. Xác minh toàn bộ bản ghi trùng lặp bằng tất cả cột, không chỉ `customer_id`.
3. Loại bản sao giống hệt nhau bằng quy tắc rõ ràng (`drop_duplicates`).
4. Kiểm tra lại trường hợp một `customer_id` còn nhiều dòng khác nhau. Nếu có, không tự ý giữ dòng đầu; cần xác định quy tắc snapshot hoặc nguồn dữ liệu.
5. Tính lại phân phối target sau khi loại duplicate.

Lý do: các bản sao trùng hoàn toàn sẽ làm tăng trọng số một số khách hàng và có thể khiến validation lạc quan nếu bản sao nằm ở các fold khác nhau.

### 5.3 Missing values

Tình trạng missing đã quan sát trước xử lý:

| Cột | Số thiếu | Tỷ lệ gần đúng |
|---|---:|---:|
| `other_bank_cards` | 2.463 | 20,32% |
| `income_month` | 1.562 | 12,89% |
| `app_logins_30d` | 1.516 | 12,51% |
| `salary_via_bank` | 718 | 5,92% |
| `occupation` | 614 | 5,07% |
| `last_contact_channel` | 515 | 4,25% |
| `age` | 396 | 3,27% |
| `gender` | 231 | 1,91% |
| `province` | 201 | 1,66% |
| `open_date` | 78 | 0,64% |

Cách xử lý đề xuất:

- Không xóa hàng loạt bản ghi chỉ vì một số feature bị thiếu.
- Với numeric: median imputation làm baseline; thêm `add_indicator`/missing indicator cho các cột thiếu đáng kể để mô hình có thể học tín hiệu từ trạng thái thiếu.
- Với categorical: tạo nhãn `Unknown` hoặc dùng imputer phù hợp.
- Fit imputer **chỉ trên training fold** trong pipeline.
- Kiểm tra missingness theo target và các phân khúc, nhưng chỉ dùng phần train trong quy trình modelling.
- Không mặc định missing có nghĩa là 0. Ví dụ `app_logins_30d` thiếu không nhất thiết đồng nghĩa với không đăng nhập.

### 5.4 Chuẩn hóa category

Các giá trị categorical đang có biến thể do hoa/thường, khoảng trắng và cách viết. Ví dụ:

- `gender`: `F`, `F `, `female`, `m`, `male`, `Nam`, `Nữ`;
- `occupation`: `salaried`, `SALARIED`, `Salaried`, `salaried `, `self employed`, `Self-employed`, `civil_servant`, `Civil servant`;
- `province`: các biến thể như `TP.HCM`, `TP HCM`, `Ho Chi Minh`, `Hồ Chí Minh`, `tp.hcm`, `HCM`; `Hà Nội`, `HN`, `Ha Noi`;
- `currency`: `VND`, `vnd`.

Quy trình:

1. Trim khoảng trắng đầu/cuối, chuẩn hóa Unicode và hoa/thường theo loại cột.
2. Xây bảng ánh xạ tường minh cho alias đã xác minh; lưu bảng map cùng source code.
3. Không gộp hai giá trị chỉ dựa trên cảm giác giống nhau. Cần đảm bảo bí danh thực sự chỉ cùng một category.
4. Giữ `none` ở `last_contact_channel` là một category có ý nghĩa; không tự động chuyển `none` thành missing.
5. Sau chuẩn hóa, kiểm tra cardinality và các category hiếm; gom nhóm `Other` chỉ nếu hợp lý và được fit/áp dụng nhất quán.

### 5.5 Audit giá trị bất thường và đơn vị

Một số giá trị quan sát trực tiếp cần kiểm tra nghiệp vụ:

- `age`: min = 0, max = 999; có 9 bản ghi bằng 999, 40 bản ghi trên 100 và 29 bản ghi dưới 18.
- `income_month`: median khoảng 17,7 nhưng max khoảng 50.500.000; có 106 bản ghi lớn hơn 1.000.
- `credit_limit`: median khoảng 40 nhưng max khoảng 385.000.000; có 247 bản ghi lớn hơn 1.000.
- `utilization_6m`: min = -1, max = 15; có giá trị âm và trên 1.
- `txn_amount_3m`: có 170 giá trị âm.
- `app_logins_30d` có các giá trị cao bất thường; `txn_count_3m` cũng có đuôi phân phối lớn.

Các thống kê này gợi ý khả năng có sentinel, lỗi nhập liệu hoặc khác biệt đơn vị, nhưng **không đủ để kết luận giá trị nào phải xóa**. Cần:

1. Xem histogram, quantile, các hàng cực trị và quan hệ giữa biến liên quan.
2. Làm rõ đơn vị và định nghĩa biến với data owner.
3. Với giá trị xác định là sentinel/lỗi: chuyển về missing hoặc sửa theo quy tắc có tài liệu.
4. Với giá trị cực trị hợp lệ: giữ lại; thử `log1p` cho feature không âm hoặc robust transformation nếu phù hợp.
5. Chạy ablation so sánh mô hình có/không xử lý outlier trên cùng fold.
6. Không dùng ngưỡng cắt được quyết định dựa trên validation/test outcome.

### 5.6 Xử lý `open_date`

Dữ liệu có ít nhất ba định dạng ngày: `%Y-%m-%d`, `%Y%m%d`, `%d/%m/%Y`. Dùng parser theo pattern/format tường minh thay vì để thư viện tự đoán toàn bộ dữ liệu.

Features có thể tạo:

- `tenure_days = snapshot_date - open_date`;
- `tenure_months` hoặc số năm giữ thẻ;
- có thể tạo bucket tenure dựa trên binning được xác định từ train.

Snapshot theo notebook là **30/06/2026**. Cần assert `open_date <= snapshot_date`; các giá trị sau snapshot phải được đánh dấu để điều tra. 78 giá trị `open_date` đang thiếu cần được xử lý như missing; không thay bằng ngày trung bình một cách tùy tiện.

---

## 6. Phase 2 — EDA và giả thuyết nghiệp vụ

### 6.1 Phân tích một biến

- Phân phối target, số lượng từng lớp và tỷ lệ churn.
- Histogram/boxplot cho numeric; phân tích cả raw scale và log scale khi phù hợp.
- Bar chart theo category, bao gồm nhóm missing/unknown.
- Missingness heatmap hoặc bảng tỷ lệ missing.
- Kiểm tra category có số lượng rất thấp.

### 6.2 Phân tích hai biến

Tập trung quan hệ feature-target bằng tỷ lệ churn và khoảng tin cậy/độ ổn định, không diễn giải tương quan như quan hệ nhân quả. Các lát cắt cần xem:

- `card_type`;
- `occupation`;
- `salary_via_bank`;
- `promo_ended_60d`;
- `utilization_6m`;
- `spend_change_pct`;
- `txn_count_3m` và `txn_amount_3m`;
- `complaints_12m`, `late_payments_12m`;
- mức độ hoạt động app và tenure.

Các phân tích sơ bộ trong file cho thấy tỷ lệ churn khác nhau giữa một số nhóm (ví dụ `promo_ended_60d`, `card_type`, `occupation`), nhưng đây chỉ là thống kê mô tả ban đầu, bị ảnh hưởng bởi duplicate và category không chuẩn hóa. Phải tính lại sau làm sạch, đồng thời tránh kết luận nhân quả.

### 6.3 Kiểm tra data leakage

Vì mục tiêu là dự đoán tại ngày 30/06/2026, với từng feature phải hỏi: **feature có được biết tại thời điểm này không?**

- Đảm bảo các số liệu `*_3m`, `*_6m`, `*_12m` tổng hợp từ giai đoạn kết thúc không muộn hơn snapshot.
- Xác minh `promo_ended_60d` và `last_contact_channel` được ghi nhận trước snapshot.
- Không sử dụng ngày đóng thẻ, trạng thái tài khoản hoặc các trường được cập nhật sau snapshot.
- Không sử dụng `churn_3m` để tạo bất cứ thống kê/encoding feature nào.
- Không dùng `customer_id` như tín hiệu dự đoán mặc định.
- Mọi thống kê từ dữ liệu phải được fit trong từng fold hoặc chỉ dựa trên training set.

**Deliverables:** EDA report, data dictionary cập nhật, bảng vấn đề chất lượng dữ liệu, quyết định xử lý có người chịu trách nhiệm và bằng chứng.

---

## 7. Phase 3 — Feature engineering

Chỉ tạo feature có ý nghĩa và có sẵn tại snapshot. Mỗi feature mới phải được kiểm tra bằng cross-validation, không mặc định rằng thêm feature sẽ giúp mô hình tốt hơn.

### 7.1 Feature đề xuất

1. **Tenure:** `tenure_days`, `tenure_months` từ `open_date`.
2. **Chỉ số giao dịch:** giữ `txn_count_3m`, `txn_amount_3m`, `spend_change_pct`; cân nhắc log-transform nếu miền giá trị và ý nghĩa nghiệp vụ cho phép.
3. **Tỷ lệ sử dụng hạn mức:** giữ `utilization_6m` và kiểm tra tính nhất quán với `credit_limit`/giao dịch. Chỉ tạo thêm tỷ lệ mới khi định nghĩa, đơn vị phù hợp.
4. **Engagement:** `app_logins_30d`, `last_contact_channel`, `complaints_12m`, `late_payments_12m`.
5. **Retention/context:** `promo_ended_60d`, `annual_fee_waived`, `card_type`, `salary_via_bank`.
6. **Missing indicators:** đặc biệt cho `income_month`, `app_logins_30d`, `other_bank_cards` và `open_date`.
7. **Interactions có giả thuyết:** ví dụ thay đổi chi tiêu × chương trình ưu đãi kết thúc; tenure × card type; sử dụng hạn mức × trễ hạn. Chỉ giữ tương tác khi validation chứng minh có ích.

### 7.2 Nguyên tắc

- Tách code tạo feature thành hàm tái sử dụng cho train và test.
- Không tính feature theo toàn bộ train + test có dùng target hoặc thông tin tương lai.
- Không target-encode categorical ngoài quy trình cross-fitting.
- Không scale numeric cho các mô hình cây nếu không cần; Logistic Regression cần scale numeric.
- Giữ lại tập feature gốc và chạy ablation để đo tác động của từng nhóm.

---

## 8. Phase 4 — Thiết kế validation và metric

### 8.1 Phương pháp chia dữ liệu

Dùng ba tập đã tạo trong `classification_splits/`:

- **Train:** 8.400 dòng (70%, 984 churners). Fit pipeline; dùng 5-fold `StratifiedKFold(shuffle=True, random_state=42)` chỉ bên trong tập này cho CV/tuning.
- **Validation:** 1.800 dòng (15%, 211 churners). Dùng để so sánh shortlist mô hình, kiểm tra calibration và chọn threshold nếu có tiêu chí nghiệp vụ. Không lặp tuning vô hạn trên validation.
- **Test nội bộ:** 1.800 dòng (15%, 210 churners). Giữ kín đến khi đã khóa preprocessing, feature set, mô hình và quy tắc threshold/calibration; báo cáo kết quả cuối cùng một lần.

Cách chia được thực hiện **sau khi loại 120 exact duplicates**, theo `stratify=churn_3m`, `random_state=42`. Tỷ lệ churn lần lượt là 11,714% / 11,722% / 11,667%, cho thấy stratification giữ phân phối lớp khá đồng đều. Không oversample/undersample trước khi chia; mọi resampling nếu có phải nằm bên trong training fold. Fit imputer, encoder, scaler và feature selection chỉ trên phần fit của từng fold.

Dataset hiện tại có một snapshot cho từng khách hàng sau de-dup, vì thế stratified random split là lựa chọn phù hợp cho holdout nội bộ. Nếu bổ sung nhiều snapshot trên cùng khách hàng hoặc mục tiêu chuyển sang dự báo theo thời gian, cần chuyển sang group split theo khách hàng hoặc temporal split tùy data contract. `classification_splits/test.csv` là test có nhãn nội bộ và không thay thế file `test.csv` chính thức chưa được cung cấp.

### 8.2 Metric

| Metric | Vai trò | Lý do |
|---|---|---|
| **ROC AUC** | Metric chính | Phù hợp yêu cầu trong notebook; đo khả năng xếp hạng churners cao hơn non-churners |
| Average Precision / PR AUC | Metric bổ sung | Hữu ích khi lớp churn chiếm tỷ lệ thấp |
| Precision, Recall, F1 | Chẩn đoán tại threshold | Phụ thuộc threshold, có ích khi xác định quy trình can thiệp |
| Recall@Top-K / Lift@Top-K | Giá trị chiến dịch | Cho biết trong nhóm khách hàng được ưu tiên có bao nhiêu churners; chọn K theo năng lực chăm sóc |
| Brier score / calibration curve | Nếu dùng xác suất thực tế | Kiểm tra xác suất dự đoán có hiệu chỉnh tốt không |

ROC AUC không phụ thuộc vào threshold cố định. Không tối ưu accuracy làm mục tiêu chính. Không so sánh mô hình chỉ bằng điểm trên leaderboard; dùng OOF/cross-validation để có ước lượng ổn định hơn.

### 8.3 Báo cáo validation

Mỗi mô hình cần có:

- AUC từng fold bên trong train, mean và standard deviation;
- ROC AUC/PR AUC trên validation;
- Kết quả test nội bộ chỉ sau khi khóa mô hình, được báo cáo riêng và không dùng để lựa chọn mô hình;
- PR AUC trung bình;
- confusion matrix, precision/recall tại threshold được chọn sau đó;
- ROC curve và precision-recall curve;
- lift/recall theo top 5%, 10% hoặc theo ngưỡng mà bộ phận kinh doanh có thể xử lý;
- OOF prediction để so sánh mô hình, calibration và chọn threshold.

---

## 9. Phase 5 — Baseline và mô hình ứng viên

### 9.1 Baseline bắt buộc

**Baseline A — constant/naive score**

- Gán cùng một score cho mọi khách hàng chỉ nhằm xác minh pipeline metric/format; ROC AUC kỳ vọng tương đương random-ranking, không phải mô hình sử dụng được.
- Giá trị `0.12` trong `sample_submission(1).csv` là giá trị mẫu cố định để minh họa format; không coi đó là dự đoán đã học hoặc benchmark AUC.

**Baseline B — Logistic Regression**

- Numeric: median imputation + scaling.
- Categorical: impute `Unknown` + one-hot encoding.
- Dùng `Pipeline` + `ColumnTransformer`, để toàn bộ preprocessing được fit bên trong từng fold.
- Khởi đầu với logistic regression không class weighting; sau đó so sánh `class_weight='balanced'` hoặc weights được tune.
- Chọn regularization thông qua cross-validation.

Notebook mẫu dùng Logistic Regression và chỉ lấy numeric features. Đây là điểm bắt đầu tốt cho baseline, nhưng kế hoạch triển khai nên đưa categorical/date vào pipeline và tránh việc tính median trên toàn bộ train trước cross-validation như cách làm đơn giản trong notebook.

### 9.2 Mô hình đề xuất để so sánh

| Mô hình | Mục đích | Ghi chú |
|---|---|---|
| Logistic Regression | Baseline dễ giải thích | Có thể thiếu quan hệ phi tuyến và tương tác |
| Random Forest / ExtraTrees | Baseline phi tuyến | Kiểm tra độ bền với tương tác; cần giới hạn depth/min samples để tránh overfit |
| Gradient Boosting (HistGradientBoosting hoặc thư viện tương đương) | Mô hình cây mạnh | Cần tiền xử lý categorical phù hợp, hoặc encoding trong pipeline |
| CatBoost (nếu môi trường cho phép) | Ứng viên ưu tiên cho tabular có categorical | Có cơ chế xử lý categorical; vẫn cần kiểm soát leakage và validation đúng |
| LightGBM/XGBoost (nếu có) | Ứng viên boosting thay thế | So sánh với CatBoost/Random Forest trên đúng fold |

Không cần huấn luyện tất cả mô hình ngay lập tức. Thứ tự khuyến nghị: Logistic Regression → CatBoost/boosting → một mô hình cây đối chứng → chọn shortlist để tuning. Với cỡ dữ liệu khoảng 12.000 khách hàng, ưu tiên mô hình có chất lượng tốt nhưng dễ kiểm soát và tái lập.

### 9.3 Class imbalance

Thử nghiệm theo thứ tự:

1. Không xử lý imbalance ngoài stratified folds.
2. So sánh với class weights.
3. Nếu cần, thử undersampling/oversampling bên trong từng training fold và chỉ thực hiện trên phần train của fold.

Không mặc định oversampling sẽ cải thiện ROC AUC. Đánh giá bằng đúng metric chính và metric nghiệp vụ; tránh làm mất thông tin hoặc khiến xác suất dự đoán mất calibration.

---

## 10. Phase 6 — Hyperparameter tuning và lựa chọn mô hình

### Quy trình

1. Dùng cross-validation baseline để xác định mô hình mạnh nhất.
2. Tune một số hyperparameter quan trọng bằng `RandomizedSearchCV` hoặc Optuna nếu sẵn có; không gian tìm kiếm phải được giới hạn và ghi lại.
3. Chạy mọi mô hình trên cùng fold/seed để so sánh công bằng.
4. Lưu kết quả theo từng lần chạy, gồm cấu hình, seed, metric từng fold, thời gian và ghi chú.
5. Chọn mô hình dựa trên mean ROC AUC, độ ổn định giữa fold, PR AUC, độ phức tạp và khả năng giải thích/triển khai.
6. Nếu chênh lệch AUC nhỏ trong sai số fold, ưu tiên giải pháp đơn giản, ổn định, ít phụ thuộc vào xử lý dữ liệu mong manh.

### Cấu hình cần tune (tham khảo)

- **Logistic Regression:** `C`, penalty tương thích với solver, `class_weight`.
- **CatBoost/Gradient Boosting:** learning rate, depth/leaves, số cây/iterations, regularization, subsampling và class weights.
- **Random Forest/ExtraTrees:** số cây, `max_depth`, `min_samples_leaf`, max_features.

Early stopping phải dùng validation fold phù hợp. Không chọn hyperparameter theo tập test hoặc dùng leaderboard như validation duy nhất.

---

## 11. Phase 7 — Threshold, calibration và quyết định kinh doanh

Vì notebook yêu cầu ROC AUC, giai đoạn chọn mô hình cần tập trung vào ranking. Threshold chỉ được quyết định sau khi đã chọn mô hình và hiểu chi phí kinh doanh.

### Nếu tổ chức cần nhãn can thiệp 0/1

- Xác định năng lực chiến dịch, ví dụ số lượng khách hàng có thể gọi/khuyến khích mỗi tuần.
- Chọn threshold bằng OOF prediction trên validation, tối ưu mục tiêu được stakeholder duyệt (ví dụ precision tối thiểu, recall mục tiêu, hoặc lợi ích kỳ vọng).
- Không dùng test labels để lựa chọn threshold.
- Báo cáo trade-off precision/recall và số khách hàng được chọn.

### Nếu xác suất được dùng như risk score

- Kiểm tra calibration curve và Brier score.
- Nếu calibration kém và xác suất có ý nghĩa vận hành, thử sigmoid/Platt scaling hoặc isotonic calibration trên dữ liệu validation/OOF phù hợp.
- Sau calibration phải đánh giá lại; không kỳ vọng calibration sẽ làm ROC AUC tăng.
- Đảm bảo người dùng hiểu rằng dự đoán là xác suất ước lượng, không phải khẳng định chắc chắn khách hàng sẽ rời đi.

---

## 12. Phase 8 — Model interpretation và error analysis

### 12.1 Giải thích mô hình

- Logistic Regression: hệ số sau preprocessing, diễn giải theo chiều tác động có điều kiện; không diễn giải hệ số thành quan hệ nhân quả.
- Mô hình cây/boosting: global feature importance và permutation importance trên validation.
- Dùng SHAP nếu thư viện phù hợp; báo cáo cả tầm quan trọng toàn cục và ví dụ cục bộ đã ẩn thông tin định danh.
- Kiểm tra độ ổn định của feature importance qua nhiều fold.

### 12.2 Phân tích lỗi

Với OOF prediction, kiểm tra:

- false negative: khách hàng churn nhưng score thấp;
- false positive: khách hàng không churn nhưng score cao;
- hiệu năng theo `card_type`, `occupation`, `province`, `gender`, nhóm tenure và nhóm missingness;
- độ ổn định trên nhóm có ít dữ liệu;
- lỗi tập trung ở outlier hoặc category chưa chuẩn hóa hay không.

Các nhóm demographic chỉ dùng cho kiểm tra chất lượng và đánh giá công bằng phù hợp với quy định tổ chức; không mặc định rằng nên dùng nhóm nhạy cảm để quyết định hành động với khách hàng. Không chia sẻ ví dụ có thông tin nhận dạng khách hàng.

---

## 13. Phase 9 — Huấn luyện cuối cùng và tạo submission

### 13.1 Trước khi dự đoán

- Nhận file `test.csv` đúng phiên bản và xác nhận schema; file này chưa có trong những file được kiểm tra ở lượt hiện tại.
- Load template `sample_submission(1).csv` hoặc bản đổi tên `sample_submission.csv`.
- Xác nhận test ID có cùng phạm vi/thứ tự mong đợi với template; template có 8.000 ID, nên số dòng test cần được xác minh trước khi kết luận đầu ra phải có đúng 8.000 dòng.
- Dùng chính pipeline preprocessing/feature engineering đã validation; không viết một bộ xử lý riêng cho test.
- Huấn luyện mô hình cuối cùng trên toàn bộ train đã xử lý sau khi đã khóa cấu hình.

### 13.2 Output contract

Tạo `submission.csv` với chính xác hai cột:

```csv
customer_id,churn_3m
CUSXXXXXXX,0.1234
```

Trong đó `churn_3m` phải là score/xác suất churn class `1` — dùng `predict_proba(X_test)[:, 1]` hoặc API xác suất tương đương. Không dùng `predict()` nếu format yêu cầu xác suất.

### 13.3 Kiểm tra tự động

- Đúng tên cột và thứ tự cột.
- Số dòng đúng với template/test đã xác nhận.
- Không có `customer_id` null hoặc duplicate nếu contract yêu cầu duy nhất.
- ID của output khớp template, theo đúng thứ tự nếu được yêu cầu.
- Không có xác suất null/NaN/inf hoặc ngoài `[0, 1]`.
- File CSV đọc lại được, encoding và delimiter đúng.
- Lưu model artifact, config, metric validation, feature list và commit/code version đi kèm submission.

**Không thể đánh giá ROC AUC trên test khi test không có target**. Chỉ tính metric trên validation/OOF; test chỉ phục vụ sinh dự đoán cuối cùng.

---

## 14. Phase 10 — Testing, reproducibility và triển khai

### 14.1 Unit/data tests

Viết kiểm thử cho:

- schema và target validity;
- deduplication;
- parser ngày với từng định dạng;
- category normalization và alias mapping;
- kiểm tra miền giá trị;
- train/test feature alignment sau preprocessing;
- pipeline không trả missing cho mô hình yêu cầu input hoàn chỉnh;
- output submission contract.

### 14.2 Reproducibility

- Cố định seed của CV và mô hình.
- Ghi phiên bản Python, pandas, scikit-learn và thư viện boosting.
- Lưu `best_params`, fold scores và thời điểm train.
- Không sửa tay submission sau khi chạy; tạo lại bằng pipeline có thể tái lập.

### 14.3 Nếu đưa vào môi trường vận hành

- Theo dõi tỷ lệ missing, category mới, phân phối score và drift feature.
- Khi nhãn churn thực tế có sẵn, theo dõi ROC AUC/PR AUC, calibration và hiệu quả chiến dịch theo thời gian.
- Xác định chu kỳ retraining và quy trình rollback.
- Quản lý quyền truy cập dữ liệu khách hàng, ẩn danh trong log/báo cáo và tuân thủ chính sách bảo mật dữ liệu của tổ chức.
- Theo dõi drift không đồng nghĩa với tự động retrain; cần tiêu chí rõ ràng và phê duyệt.

---

## 15. Rủi ro chính và phương án giảm thiểu

| Rủi ro | Tác động | Giảm thiểu |
|---|---|---|
| 120 dòng trùng lặp hoàn toàn | Metric lệch và tăng trọng số một số khách hàng | Dedup trước CV, lưu log trước/sau xử lý |
| Target imbalance | Accuracy dễ gây hiểu sai | ROC AUC là metric chính; thêm PR AUC/lift/recall |
| Category không chuẩn hóa | Một category bị tách thành nhiều nhóm nhỏ | Alias map có kiểm chứng, chuẩn hóa nhất quán |
| Thiếu dữ liệu | Mất tín hiệu hoặc sai lệch | Imputation trong pipeline, missing indicators, phân tích missingness |
| Outlier/đơn vị không nhất quán | Mô hình bị chi phối bởi cực trị | Xác minh với data owner, robust treatment và ablation |
| Parse ngày sai | Tenure sai hoặc vi phạm snapshot | Parse theo format tường minh và unit test |
| Leakage từ feature sau snapshot | Validation tốt giả tạo | Audit thời điểm có sẵn của từng biến và feature lineage |
| Validation ngẫu nhiên quá lạc quan khi có thời gian/khách hàng lặp | Khả năng generalize tương lai kém | Group/time split nếu cấu trúc dataset thể hiện điều đó |
| Tuning quá mức | AUC validation không ổn định | Dùng cùng folds, giới hạn search, ưu tiên mô hình ổn định |
| Thiếu `test.csv` | Chưa thể tạo submission cuối cùng | Yêu cầu bổ sung test và kiểm tra ID với template |
| Xác suất không calibration | Diễn giải risk score sai | Đánh giá calibration nếu xác suất được dùng để ra quyết định |

---

## 16. Kế hoạch thời gian và deliverables

Kế hoạch tham khảo cho một vòng triển khai ban đầu; thời lượng thực tế phụ thuộc tốc độ làm rõ data quality và hạ tầng.

| Giai đoạn | Công việc | Deliverable / cổng nghiệm thu |
|---|---|---|
| Ngày 1 | Setup, kiểm tra schema, target, duplicate, missing | Data audit report và data contract |
| Ngày 2–3 | EDA, điều tra giá trị bất thường, chuẩn hóa ngày/category | EDA report, quyết định cleaning có ghi nhận |
| Ngày 4 | Baseline Logistic Regression với pipeline đầy đủ | Baseline CV metrics, OOF predictions |
| Ngày 5–6 | Feature engineering và thử nghiệm mô hình boosting/cây | Bảng so sánh mô hình trên cùng folds |
| Ngày 7 | Tune shortlist và phân tích stability | Best config, fold scores, PR AUC |
| Ngày 8 | Error analysis, feature importance, calibration/threshold nếu cần | Model card và đề xuất vận hành |
| Ngày 9 | Fit final model, nhận test và tạo submission | File submission đã qua automated checks |
| Ngày 10 | Kiểm thử, hoàn thiện tài liệu và reproducibility | README, tests, artifact/config, nghiệm thu |

---

## 17. Checklist nghiệm thu cuối cùng

### Data

- [ ] Đúng nguồn dữ liệu churn; không nhầm với dataset khác trong workspace.
- [ ] Dedupe và kiểm tra `customer_id` được thực hiện trước khi chia tập/CV.
- [ ] Đã tạo train/val/test 70/15/15 với số dòng 8.400/1.800/1.800 và seed 42.
- [ ] Phân phối target được tính lại sau cleaning.
- [ ] Missingness, category aliases, giá trị cực trị và định dạng ngày đã được kiểm tra.
- [ ] Các feature đều có sẵn tại 30/06/2026.

### Model/evaluation

- [ ] Pipeline preprocessing nằm bên trong cross-validation.
- [ ] ROC AUC được dùng làm metric chính.
- [ ] Có mean/std ROC AUC và kết quả từng fold trên train.
- [ ] Validation được dùng cho model selection; test nội bộ chỉ được chấm một lần sau khi khóa pipeline.
- [ ] Có PR AUC hoặc metric bổ sung để đánh giá lớp churn ít hơn.
- [ ] Mô hình được so sánh trên cùng các folds.
- [ ] Không tuning dựa vào test labels/leaderboard như nguồn đánh giá duy nhất.
- [ ] Có error analysis và giải thích mô hình phù hợp.

### Submission/reproducibility

- [ ] Đã nhận `test.csv` chính thức và xác minh schema.
- [ ] Dùng xác suất class `1`, không phải nhãn dự đoán rời rạc.
- [ ] ID/độ dài/cột output khớp template đã xác nhận.
- [ ] Không có NaN, Inf hoặc score ngoài `[0, 1]`.
- [ ] Lưu code/config/seed/model artifact/validation report.

---

## 18. Kết luận và khuyến nghị thực thi

Bắt đầu bằng một pipeline sạch và baseline có thể tái lập, không bắt đầu bằng tuning phức tạp. Đối với dataset này, các bước có khả năng ảnh hưởng đáng kể đến chất lượng đánh giá là xử lý 120 dòng duplicate, chuẩn hóa categorical, kiểm tra các giá trị cực trị/đơn vị, parse `open_date` tường minh và đảm bảo không có leakage từ sau snapshot.

Lộ trình khuyến nghị:

1. Deduplicate và hoàn thành data-quality audit.
2. Tạo `ColumnTransformer` + `Pipeline` gồm numeric imputation/scaling và categorical imputation/one-hot encoding.
3. Thiết lập train/validation/test 70/15/15 cố định; chạy 5-fold Stratified CV chỉ trên train và lưu OOF predictions.
4. Dùng Logistic Regression làm baseline; so sánh với CatBoost hoặc gradient boosting.
5. Chọn mô hình bằng CV trong train và validation; sau khi khóa mô hình, đánh giá test nội bộ một lần, dùng PR AUC và lift để hiểu giá trị vận hành.
6. Khi cần submission benchmark, fit pipeline đã chốt trên toàn bộ dataset gán nhãn phù hợp rồi dự đoán `test.csv` chính thức (cần nhận file này riêng) và kiểm tra submission.

**Phụ thuộc còn thiếu:** `test.csv` chưa nằm trong các file được kiểm tra ở lượt này. Có thể hoàn tất EDA và validation trên train trước, nhưng chỉ tạo được submission cuối cùng khi nhận đúng test set.

