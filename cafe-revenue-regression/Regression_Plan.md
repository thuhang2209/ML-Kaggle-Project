# Kế hoạch triển khai dự án Regression — Dự báo doanh thu cửa hàng cà phê

> **Vai trò:** Senior Data Scientist  
> **Bài toán:** Dự báo doanh thu theo ngày cho từng cửa hàng (*panel time-series regression*)  
> **Target:** `revenue` — doanh thu ngày, đơn vị triệu VND  
> **Metric chính:** RMSLE (Root Mean Squared Logarithmic Error)  
> **Giai đoạn dự báo theo notebook:** 01/01/2026–31/03/2026

---

## 1. Tóm tắt dự án

Mục tiêu là xây dựng một mô hình hồi quy dự báo doanh thu hằng ngày cho từng cửa hàng cà phê trong giai đoạn tương lai. Dữ liệu lịch sử nằm trong `train.csv`; thông tin cửa hàng, thời tiết và ngày lễ nằm trong các bảng bổ trợ. Đây không phải bài toán regression thông thường với các dòng độc lập: mỗi cửa hàng có chuỗi quan sát theo thời gian, doanh thu chịu ảnh hưởng của thứ trong tuần, mùa vụ, khuyến mãi, đặc điểm cửa hàng, thời tiết và các sự kiện đặc biệt.

Notebook `starter_cafe.ipynb` định nghĩa bài toán và cung cấp baseline:

- Dự báo doanh thu cho từng cửa hàng trong khoảng 01/01/2026–31/03/2026 (Quý 1/2026).
- Đánh giá bằng **RMSLE** trên doanh thu không âm.
- Baseline hiện có lấy **median doanh thu theo cặp `(store_id, weekday)`**, dự phòng bằng median của cửa hàng rồi median toàn bộ dữ liệu.
- Notebook ghi nhận **RMSLE = 0.2387** khi huấn luyện trên lịch sử trước 01/10/2025 và đánh giá trên Quý 4/2025 (01/10/2025 – 31/12/2025). Đây là mốc benchmark chuẩn mực cho chiến lược Time-based Out-Of-Time (OOT) validation của dự án.

### Kết quả kỳ vọng

1. Một pipeline dữ liệu có thể chạy lại và không làm rò rỉ thông tin từ tương lai.
2. Một mô hình đạt kết quả tốt hơn baseline trên validation theo thời gian (RMSLE < 0.2387).
3. Báo cáo giải thích sai số, rủi ro dữ liệu và hiệu năng theo nhóm cửa hàng.
4. Khi nhận được file `test.csv` chính thức từ giảng viên, xuất đúng file `submission.csv` gồm hai cột `id`, `revenue`, giữ đúng thứ tự `id` trong test.

**Lưu ý về tập test:** File `test.csv` chính thức (gồm 3.869 dòng dự báo cho Quý 1/2026 từ 01/01/2026 đến 31/03/2026) sẽ được giảng viên tải lên Kaggle sau. Do đó, toàn bộ dữ liệu hiện có trong `train.csv` (sau khi deduplicate và làm sạch) sẽ được bảo toàn làm **Tập huấn luyện đầy đủ (Full Training Set)**; tuyệt đối **không cắt xẻ `train.csv` thành 3 file vật lý cố định trên đĩa**. Thay vào đó, nhóm thiết lập chiến lược kiểm định chéo theo thời gian (Time-based Out-of-Time Validation) trực tiếp trong mã nguồn để phục vụ việc lựa chọn mô hình và tinh chỉnh siêu tham số.

---

## 2. Hiểu dữ liệu hiện có

### 2.1. Các nguồn dữ liệu

| File | Nội dung | Cột chính | Lưu ý triển khai |
|---|---|---|---|
| `train.csv` | Doanh thu lịch sử theo cửa hàng/ngày | `date`, `store_id`, `is_promo`, `revenue` | Target nằm ở `revenue`; cần kiểm tra khóa ngày-cửa hàng trùng lặp và ngoại lệ |
| `stores.csv` | Đặc điểm cửa hàng | `store_id`, `city`, `district`, `store_type`, `area_m2`, `seats`, `has_parking`, `open_date` | Có giá trị thiếu ở `area_m2`; `store_type` không thống nhất kiểu chữ; `open_date` có nhiều định dạng |
| `weather.csv` | Thời tiết theo thành phố/ngày | `date`, `city`, `temp_c`, `rain_mm` | Có giá trị thiếu và cần kiểm tra phạm vi hợp lý của nhiệt độ/lượng mưa |
| `holidays.csv` | Ngày lễ/sự kiện | `date`, `holiday_name`, `type` | Nhiều hơn một bản ghi có thể nằm trên cùng một ngày; phải tổng hợp trước khi join |
| `sample_submission.csv` | Mẫu kết quả nộp | `id`, `revenue` | Có 3.869 ID từ 100000 đến 103868; không chứa biến giải thích |
| `starter_cafe.ipynb` | Notebook khởi đầu | Baseline, cách chia validation, mô tả metric | Dùng làm điểm xuất phát, không coi baseline là mô hình cuối cùng |
| `test.csv` | Dữ liệu dự báo | Chưa có trong các file được cung cấp lần này | Bắt buộc có để ánh xạ dòng dự báo và tạo submission đúng |

### 2.2. Các thống kê đã quan sát

| Hạng mục | Quan sát | Ý nghĩa/rủi ro |
|---|---|---|
| `train.csv` | 26.632 dòng, từ 01/01/2024 đến 31/12/2025 | Có khoảng hai năm lịch sử |
| Số cửa hàng | 45 cửa hàng | Có thể dùng đặc điểm cửa hàng và mô hình hóa khác biệt theo cửa hàng |
| Số ngày khác nhau trong train | 731 ngày | Dữ liệu panel không đầy đủ cho mọi cửa hàng-ngày |
| Số dòng theo cửa hàng | Từ 21 đến 727 dòng | Một số cửa hàng có lịch sử rất ngắn; cần phân tích riêng nhóm cửa hàng mới/ít quan sát |
| `revenue` | Median khoảng 18.28; phân vị 99% khoảng 44.807; max 53.600 | Phân phối lệch phải rất mạnh; một số cực trị cần điều tra |
| Giá trị `revenue` âm/0 | Không thấy giá trị âm hoặc bằng 0 trong profiling ban đầu | Vẫn cần kiểm tra sau các bước xử lý dữ liệu |
| Khóa `(date, store_id)` | Có 132 bản ghi trùng khóa và đây cũng là các bản ghi trùng hoàn toàn trong kiểm tra ban đầu | Xác minh rồi loại bản sao, không để một quan sát được tính nhiều lần |
| `stores.csv` | 45 dòng; thiếu 2 giá trị `area_m2` | Impute có kiểm soát và tạo cờ thiếu nếu cần |
| `store_type` | Có biến thể như `street`/`Street`, `office`/`Office` | Chuẩn hóa category trước khi encode hoặc đưa vào mô hình hỗ trợ categorical |
| `open_date` | Có cả chuỗi ISO `YYYY-MM-DD` và `DD/MM/YYYY` | Parse theo từng định dạng; không áp dụng một `dayfirst` chung một cách mù quáng |
| `weather.csv` | 1.593 dòng; dữ liệu đến 31/03/2026; 16 giá trị `rain_mm` thiếu | Có khoảng trống ngày-thành phố và cần xử lý thiếu |
| Thời tiết bất thường | Có giá trị `temp_c = 84.6` ở Ho Chi Minh ngày 25/06/2025 trong profiling ban đầu | Kiểm tra nguồn và đơn vị, không tự động xóa nếu chưa hiểu nguyên nhân |
| `holidays.csv` | 55 dòng, 53 ngày khác nhau; dữ liệu đến 08/03/2026 | Phạm vi chưa bao phủ toàn bộ kỳ test đến 31/03/2026; cần bổ sung/xác minh lịch ngày lễ hoặc tạo cờ coverage |

**Lưu ý về các con số:** đây là kết quả kiểm tra sơ bộ từ các file cung cấp và đầu ra đã lưu trong notebook. Cần chạy lại các kiểm tra trong pipeline để tạo báo cáo dữ liệu có thể tái lập trước khi huấn luyện chính thức.

### 2.3. Kế hoạch phân chia dữ liệu và Chiến lược Validation

Toàn bộ dữ liệu trong `train.csv` (26.632 dòng, sau khi loại bỏ 132 dòng trùng lặp hoàn toàn còn **26.500 dòng sạch**) là tập dữ liệu huấn luyện trọn vẹn 2 năm lịch sử (từ 01/01/2024 đến 31/12/2025).

Vì file **`test.csv` chính thức sẽ được giảng viên tải lên Kaggle sau**, nhóm **không chia tách `train.csv` thành 3 file vật lý (train/val/test) lưu trên đĩa**. Việc giữ nguyên tập train đầy đủ giúp:
- Tránh phân mảnh dữ liệu và lãng phí 3 tháng dữ liệu quý 4/2025 khi huấn luyện mô hình sau cùng.
- Tránh nhầm lẫn giữa tập "test tự tạo" với file `test.csv` chính thức của cuộc thi.

Thay vào đó, chiến lược kiểm định được thực hiện theo cơ chế **Time-based Out-Of-Time (OOT) Validation** trực tiếp trong code:

| Phân vùng | Quy tắc ngày | Số dòng | Tỷ lệ trên 26.500 dòng sạch | Vai trò kỹ thuật |
|---|---|---:|---:|---|
| **Tập Huấn Luyện (Train)** | `date < 2025-10-01` (01/01/2024 – 30/09/2025) | 22.668 | 85,54% | Dùng để fit pipeline, transformers, và huấn luyện mô hình |
| **Tập Kiểm Định (Validation OOT)** | `2025-10-01 <= date <= 2025-12-31` (Q4/2025) | 3.832 | 14,46% | So sánh baseline, lựa chọn feature, tinh chỉnh hyperparameter và ensemble |

**Lý do chọn Quý 4/2025 làm Validation Set**:
- Độ dài validation là 92 ngày, mô phỏng chính xác horizon dự báo 90 ngày của Quý 1/2026 (`test.csv`).
- Đảm bảo tính tuần tự thời gian tuyệt đối: Mô hình chỉ học từ quá khứ (trước tháng 10/2025) để dự đoán tương lai, loại bỏ hoàn toàn rủi ro Data Leakage.
- Khớp chuẩn với thiết lập benchmark của `starter_cafe.ipynb` (đạt RMSLE baseline = 0.2387).

Mã nguồn chia tập logic trong pipeline:

```python
import pandas as pd

# 1. Đọc dữ liệu và loại bỏ bản ghi trùng lặp hoàn toàn
source = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
source = source.drop_duplicates()  # loại 132 dòng trùng lặp

# 2. Phân chia logic theo mốc thời gian (không lưu ra nhiều file csv riêng)
train_df = source[source["date"] < "2025-10-01"].copy()
val_df = source[(source["date"] >= "2025-10-01") & (source["date"] <= "2025-12-31")].copy()

print(f"Train records: {len(train_df):,} | Val records: {len(val_df):,}")
```

**Quy trình khi nhận file `test.csv` chính thức từ giảng viên**:
1. Khóa pipeline, cấu hình feature và siêu tham số tốt nhất đã được kiểm chứng trên tập Validation (Q4/2025).
2. **Huấn luyện lại (Final Retrain)** mô hình trên **toàn bộ 100% dữ liệu lịch sử sạch** (26.500 dòng, từ 01/01/2024 đến 31/12/2025) để mô hình cập nhật xu hướng và hành vi tiêu dùng gần nhất của Quý 4/2025.
3. Sinh dự báo cho 3.869 dòng trong `test.csv` và xuất file `submission.csv`.

---

## 3. Xác định bài toán và tiêu chí thành công

### 3.1. Định nghĩa bài toán

- **Đơn vị dự báo:** một cặp `(date, store_id)` trong tập test.
- **Biến mục tiêu:** `revenue`.
- **Loại mô hình:** supervised regression; về cấu trúc dữ liệu, đây là dự báo chuỗi thời gian nhiều cửa hàng.
- **Horizon:** theo notebook là 90 ngày, từ 01/01/2026 đến 31/03/2026.
- **Ràng buộc đầu ra:** doanh thu dự đoán phải hữu hạn và không âm.

### 3.2. Metric chính: RMSLE

Công thức:

\[
\mathrm{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}
\left[\log(1+\hat{y}_i)-\log(1+y_i)\right]^2}
\]

Trong đó `y` là doanh thu thật và `ŷ` là dự báo sau khi giới hạn không âm. RMSLE giảm ảnh hưởng tuyệt đối của cửa hàng có doanh thu lớn so với RMSE thông thường và tập trung vào sai lệch tương đối/logarit. Tuy vậy, một ngoại lệ lớn vẫn cần được điều tra vì nó có thể ảnh hưởng mạnh đến chất lượng dữ liệu và kết quả đánh giá.

Đo lường trong code nên dùng cùng định nghĩa với notebook:

```python
import numpy as np

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
```

### 3.3. KPI đề xuất

- **Baseline chuẩn mực:** Baseline median `(store_id, weekday)` đạt RMSLE **0.2387** trên tập validation Q4/2025 (01/10/2025 – 31/12/2025) khi fit trên dữ liệu đến 30/09/2025.
- **Mục tiêu thử nghiệm:** Cải thiện ít nhất 15%–30% tương đối so với baseline (mục tiêu đưa RMSLE từ `0.2387` xuống dưới `0.1800`, tiệm cận `0.1632` với mô hình Ensemble LightGBM + CatBoost).
- **Độ ổn định:** Mô hình duy trì sai số ổn định qua các fold thời gian walk-forward và không bị thiên lệch theo từng cụm cửa hàng.
- **Tính đúng đắn:** Không có Data Leakage từ tương lai, không có `NaN`/`inf` trong dự đoán, giá trị dự đoán được biến đổi ngược bằng `np.expm1()` và clip chặn dưới $\ge 0$.

Metric phụ để chẩn đoán gồm MAE trên doanh thu gốc, sai số theo cửa hàng, sai số theo tháng, sai số theo loại cửa hàng, ngày khuyến mãi/ngày thường và nhóm cửa hàng có lịch sử ngắn. Không dùng các metric phụ này thay cho RMSLE khi lựa chọn mô hình chính.

---

## 4. Kế hoạch khám phá dữ liệu (EDA)

### 4.1. Phân tích target

1. Vẽ histogram và biểu đồ phân vị của `revenue` trên cả thang gốc và `log1p(revenue)`.
2. Xem phân phối theo `store_id`, tháng, năm, thứ trong tuần, `is_promo` và ngày lễ.
3. Tính median, mean, IQR, P95/P99, max theo cửa hàng; xác định cửa hàng có độ phân tán khác thường.
4. Vẽ chuỗi thời gian của từng cửa hàng và tổng hợp theo ngày để phát hiện đổi mức, mùa vụ hoặc khoảng thiếu bất thường.
5. Điều tra cụm doanh thu rất cao của `S12` trong tháng 03/2024. Các giá trị lớn nhất trong profiling ban đầu tập trung ở cửa hàng này và khác xa phân phối thông thường. Cần kiểm tra đơn vị, quy trình nhập liệu, khả năng gộp giao dịch theo ngày và tính hợp lệ của sự kiện kinh doanh.

**Nguyên tắc:** không xóa hoặc winsorize ngoại lệ chỉ vì giá trị lớn. Trước hết xác minh nguyên nhân. Nếu không thể xác minh, lưu lại cả dữ liệu gốc và phiên bản xử lý, so sánh mô hình có/không có cách xử lý ngoại lệ bằng validation theo thời gian.

### 4.2. Độ phủ và tính đầy đủ

- Đếm số bản ghi theo ngày và theo cửa hàng.
- Lập ma trận `store_id × date` để phân biệt ngày không có bản ghi, cửa hàng chưa mở và doanh thu bằng 0 (nếu có ý nghĩa nghiệp vụ).
- Không tự động gán `revenue = 0` cho mọi ngày thiếu; thiếu bản ghi không đồng nghĩa với cửa hàng không có doanh thu.
- Đối chiếu `open_date` với ngày quan sát đầu tiên của từng cửa hàng.
- Kiểm tra ID cửa hàng trong `train` đều có mặt trong `stores`; xác minh thêm test khi nhận được `test.csv`.

### 4.3. Phân tích biến giải thích

- `is_promo`: xác nhận giá trị hợp lệ và tỷ lệ khuyến mãi theo thời gian/cửa hàng.
- Store metadata: phân phối `area_m2`, `seats`, `has_parking`, các mức `city`, `district`, `store_type`.
- Weather: nhiệt độ và mưa theo thành phố/mùa; kiểm tra ngoại lệ vật lý, ngày thiếu và khóa ghép.
- Holidays: số sự kiện theo ngày, loại sự kiện, ngày lễ liên tiếp; xác định ngày nào chưa có thông tin trong tập holiday.
- Correlation hoặc feature importance chỉ dùng như công cụ khám phá, không xem tương quan là bằng chứng nhân quả.

---

## 5. Làm sạch dữ liệu và xây dựng bảng học

### 5.1. Chuẩn hóa ngày và category

1. Parse `date` ở tất cả bảng sang `datetime64[ns]`.
2. Với `stores.open_date`, xử lý theo mẫu chuỗi:
   - Nếu khớp `YYYY-MM-DD`, parse theo `%Y-%m-%d`.
   - Nếu khớp `DD/MM/YYYY`, parse theo `%d/%m/%Y`.
   - Báo lỗi/ghi log các giá trị còn lại thay vì tự suy đoán.
3. Chuẩn hóa category như `store_type`: trim khoảng trắng, chuyển về chữ thường, map các biến thể về cùng một nhãn.
4. Chuẩn hóa tên thành phố thống nhất giữa `stores` và `weather` nhưng không tự ý thay đổi giá trị gốc trong nguồn.
5. Chuyển `has_parking`, `is_promo` sang kiểu dữ liệu phù hợp; kiểm tra chúng chỉ có các giá trị hợp lệ.

Ví dụ parse `open_date` an toàn:

```python
import pandas as pd

s = stores["open_date"].astype("string").str.strip()
iso_mask = s.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
dmy_mask = s.str.fullmatch(r"\d{2}/\d{2}/\d{4}", na=False)

stores["open_date_parsed"] = pd.NaT
stores.loc[iso_mask, "open_date_parsed"] = pd.to_datetime(
    s[iso_mask], format="%Y-%m-%d", errors="coerce"
)
stores.loc[dmy_mask, "open_date_parsed"] = pd.to_datetime(
    s[dmy_mask], format="%d/%m/%Y", errors="coerce"
)
```

### 5.2. Xử lý bản ghi trùng

- Kiểm tra trùng hoàn toàn, trùng khóa và trùng có khác giá trị.
- Với dữ liệu hiện tại, profiling ban đầu ghi nhận 132 dòng trùng hoàn toàn theo `(date, store_id, is_promo, revenue)`. Nếu xác minh đó là bản sao nhập liệu, loại bản sao trước khi tạo features.
- Nếu có nhiều bản ghi cùng `(date, store_id)` nhưng revenue khác nhau, không tự động giữ dòng đầu: cần xác định đó là giao dịch, bản ghi cập nhật hay lỗi dữ liệu và định nghĩa quy tắc tổng hợp phù hợp.
- Sau xử lý, yêu cầu khóa `(date, store_id)` duy nhất nếu một dòng đại diện cho doanh thu ngày của một cửa hàng.

### 5.3. Missing values và giá trị bất thường

- `stores.area_m2`: impute bằng median phù hợp (toàn cục hoặc theo nhóm cửa hàng nếu đủ dữ liệu) và thêm cờ `area_m2_missing`.
- `weather.rain_mm`: giữ cờ `rain_missing`; impute theo phân phối thành phố/tháng hoặc phương pháp đơn giản chỉ fit trên tập train của mỗi fold.
- `weather.temp_c`: kiểm tra giá trị ngoài phạm vi hợp lý theo đơn vị và từng thành phố; điều tra điểm `84.6` trước khi quyết định sửa/loại.
- `revenue`: xác minh cực trị, dữ liệu thiếu, giá trị âm/0 và sai đơn vị. Không dùng dữ liệu validation để quyết định ngưỡng xử lý hoặc thống kê imputation.
- Fit mọi imputer/encoder/scaler trên training fold rồi chỉ transform validation/test.

### 5.4. Join dữ liệu

Bảng học nên có một dòng cho mỗi `(date, store_id)` của train hoặc test. Luồng ghép:

1. Bắt đầu từ các dòng `(date, store_id, is_promo)` trong train/test.
2. Join `stores` bằng `store_id` (many-to-one; xác nhận một dòng metadata cho mỗi ID cửa hàng).
3. Join `weather` bằng `(date, city)` sau khi xác nhận khóa này duy nhất và chuẩn hóa tên thành phố.
4. Tổng hợp `holidays` về **một dòng mỗi `date`** trước khi join, ví dụ:
   - `is_holiday`;
   - `holiday_count`;
   - `holiday_types` hoặc cờ riêng theo loại;
   - `holiday_names` nếu có ích cho mô hình hỗ trợ categorical.
5. Kiểm tra số dòng sau mỗi join không tăng ngoài dự kiến, không mất bản ghi test và không có nhân bản do quan hệ one-to-many.
6. Tạo report tỷ lệ match/unmatch của mỗi bảng bổ trợ.

**Lưu ý kỳ test:** `holidays.csv` hiện kết thúc ở 08/03/2026 nhưng kỳ dự báo theo notebook kéo dài đến 31/03/2026. Cần xác minh và bổ sung lịch ngày lễ/sự kiện cho phần còn thiếu nếu nguồn dữ liệu cho phép. Nếu không thể bổ sung, phải tạo cờ về độ phủ holiday và ghi rõ giới hạn; không nên mặc định mọi ngày không có trong bảng là ngày bình thường.

---

## 6. Feature engineering

Feature chỉ được tạo từ thông tin có thể biết tại thời điểm dự báo. Mỗi feature cần có định nghĩa rõ và kiểm tra được leakage.

### 6.1. Calendar features

- `year`, `quarter`, `month`, `day`, `day_of_week`, `day_of_year`, `week_of_year`.
- `is_weekend`, `is_month_start`, `is_month_end`, `is_month_mid` nếu có lý do nghiệp vụ.
- Chu kỳ sin/cos cho thứ trong tuần và ngày trong năm, nếu dùng mô hình tuyến tính.
- Cờ giao mùa hoặc ngày đặc biệt nếu giúp mô hình nhưng phải được xác nhận từ lịch.
- Với Tết và các ngày lễ di động, dùng lịch holiday đáng tin cậy thay vì chỉ dựa vào tháng dương lịch.

### 6.2. Store features

- `store_id` như categorical feature, không mặc định xem mã `S01`, `S02`… là đại lượng liên tục.
- `city`, `district`, `store_type` đã chuẩn hóa.
- `area_m2`, `seats`, `has_parking` và cờ thiếu.
- `open_date_parsed`, `store_age_days = date - open_date_parsed`, `store_age_months` hoặc nhóm vòng đời cửa hàng.
- Cờ `new_store`/`short_history` dựa trên số quan sát lịch sử tại thời điểm train, không dựa vào thông tin tương lai.
- Nếu số mẫu theo một nhóm category quá nhỏ, dựa vào regularization hoặc gom nhóm có lý do; không tạo nhóm chỉ để khớp noise của validation.

### 6.3. Khuyến mãi, thời tiết và ngày lễ

- `is_promo` nếu biến này thực sự có sẵn cho toàn bộ test tại thời điểm phát dự báo.
- `temp_c`, `rain_mm`, cờ mưa (`rain_mm > 0`), và cờ missing weather.
- Có thể thử nonlinear features như bins nhiệt độ/mưa hoặc tương tác thời tiết × loại cửa hàng, nhưng chỉ giữ nếu cross-validation chứng minh có ích.
- Holiday features sau khi tổng hợp: `is_holiday`, `holiday_count`, loại ngày lễ, số ngày cách tới/qua ngày lễ nếu nguồn lịch bao phủ đầy đủ.
- Không dùng thời tiết quan sát thực tế trong tương lai ở môi trường production nếu thời điểm dự báo chưa có dữ liệu đó. Trong thiết lập cuộc thi, chỉ dùng thời tiết tương lai nếu bộ dữ liệu chính thức chủ đích cung cấp và quy tắc bài toán cho phép; trong triển khai thực tế cần dùng dự báo thời tiết hoặc chạy lại forecast khi có quan sát mới.

### 6.4. Historical / target-derived features

Có thể thử:

- Median/mean doanh thu lịch sử theo `(store_id, weekday)`, `(store_id, month)`, `(store_type, weekday)`.
- Lags 1, 7, 14, 28 ngày; rolling median/mean 7/14/28 ngày; độ lệch so với median lịch sử.
- Số ngày kể từ lần quan sát gần nhất, số lượng ngày lịch sử đã có, median doanh thu lịch sử của cửa hàng.
- Target encoding cho category chỉ thực hiện theo cơ chế out-of-fold hoặc tính hoàn toàn từ phần history của mỗi fold.

**Quy tắc chống leakage cho dự báo 90 ngày:** không tạo lag/rolling feature của validation/test bằng doanh thu thật của các ngày validation/test. Khi mô phỏng dự báo batch nhiều ngày:

- Dùng dữ liệu quan sát được đến cutoff ban đầu.
- Sau ngày đầu tiên trong horizon, nếu mô hình cần lags, dùng dự đoán đã tạo cho các ngày trước trong horizon (recursive forecasting), hoặc chọn phương án không dùng lags dài hạn.
- Validation phải mô phỏng đúng cách dự báo sẽ vận hành khi test có cả 90 ngày tương lai. Không được dùng target thật của ngày 1/10 để tạo lag cho dự đoán ngày 2/10 nếu trong tình huống test không có doanh thu thật ngày 1/10.

### 6.5. Baseline feature set

Để có đường cơ sở đáng tin cậy, triển khai theo thứ tự:

1. Median `(store_id, weekday)`.
2. Thêm calendar + store metadata.
3. Thêm `is_promo`, holiday và weather.
4. Thêm target aggregates/lags, nhưng chỉ khi validation time-series cho thấy cải thiện ổn định.

---

## 7. Chiến lược chia tập và đánh giá

### 7.1. Phân vai Train / Validation theo thời gian (Time-based Out-Of-Time Split)

Vì file `test.csv` chính thức sẽ được giảng viên tải lên Kaggle sau, quy trình kiểm định nội bộ được tổ chức hoàn toàn bằng cơ chế phân chia theo thời gian trong mã nguồn:

- **Tập Train (Quá khứ):** `date < 2025-10-01` (01/01/2024 – 30/09/2025) — **22.668 dòng** (85,54% dữ liệu sạch). Toàn bộ transformers, feature encoders, scalers và mô hình chỉ được fit trên giai đoạn này.
- **Tập Validation (Out-Of-Time):** `2025-10-01 <= date <= 2025-12-31` (Quý 4/2025) — **3.832 dòng** (14,46% dữ liệu sạch). Dùng để đánh giá độ chính xác thực tế, chọn đặc trưng, tinh chỉnh siêu tham số và xác định trọng số Ensemble.

**Lợi thế kỹ thuật**:
- Quý 4/2025 có độ dài 92 ngày, tương đồng với horizon 90 ngày của tập test chính thức (Quý 1/2026).
- Tái lập chuẩn xác điểm RMSLE baseline `0.2387` của notebook khởi đầu `starter_cafe.ipynb`.
- Tuyệt đối không xáo trộn dòng ngẫu nhiên (no random shuffle) nhằm bảo vệ tính đơn điệu thời gian và ngăn chặn data leakage.

### 7.2. Walk-forward Validation (Expanding Window CV)

Để kiểm tra xem mô hình có hoạt động ổn định qua các mùa khác nhau trong năm hay không, nhóm thiết lập các fold tịnh tiến thời gian bổ sung:

| Fold | Giai đoạn Huấn luyện (Train) | Giai đoạn Đánh giá (Val) | Mục đích kiểm định |
|---|---|---|---|
| **Fold 1** | Đến 31/12/2024 (1 năm) | 01/01/2025 – 31/03/2025 (Q1/2025) | Đo năng lực dự báo quý 1 (có Tết Âm Lịch 2025) |
| **Fold 2** | Đến 31/03/2025 | 01/04/2025 – 30/06/2025 (Q2/2025) | Kiểm tra ổn định đầu mùa hè |
| **Fold 3** | Đến 30/06/2025 | 01/07/2025 – 30/09/2025 (Q3/2025) | Kiểm tra giai đoạn mùa mưa và cao điểm |
| **Primary Val (OOT)** | Đến 30/09/2025 (21 tháng) | 01/10/2025 – 31/12/2025 (Q4/2025) | **Fold chính để chốt mô hình và trọng số Ensemble** |

Mọi fold đều tuân thủ nguyên tắc: chỉ học từ quá khứ, đánh giá ở tương lai. Báo cáo đánh giá ghi nhận RMSLE từng fold để đảm bảo mô hình không overfit vào riêng một mùa cụ thể.

### 7.3. Đánh giá theo nhóm

Báo cáo RMSLE hoặc sai số log trung bình theo:

- Cửa hàng, thành phố, loại cửa hàng.
- Tháng và ngày trong tuần.
- Có/không khuyến mãi; có/không ngày lễ.
- Cửa hàng có nhiều lịch sử và cửa hàng mới/ít quan sát.
- Nhóm doanh thu thấp, trung bình, cao.

Với nhóm ít mẫu, hiển thị cả số lượng quan sát; tránh kết luận từ các nhóm quá nhỏ.

---

## 8. Lộ trình mô hình hóa

### Giai đoạn 1 — Baseline có thể tái lập

- Chạy lại median `(store_id, weekday)` và các fallback giống notebook.
- Lưu dự đoán validation, metric và phiên bản dữ liệu.
- Xác nhận xử lý duplicate có nhất quán; nếu baseline được tái tạo trên dữ liệu đã deduplicate, ghi lại metric mới thay vì so sánh không cùng điều kiện.

### Giai đoạn 2 — Mô hình tuyến tính

**Ridge/ElasticNet trên `log1p(revenue)`** với one-hot encoding cho category và chuẩn hóa numeric features. Ưu điểm: nhanh, dễ kiểm tra, tạo benchmark có khả năng giải thích. Dùng pipeline để đảm bảo imputer/encoder chỉ fit trên fold train.

### Giai đoạn 3 — Tree-based models

Thử nghiệm trên cùng bộ feature và split:

- Random Forest / Extra Trees: baseline phi tuyến, ít yêu cầu scale.
- HistGradientBoosting hoặc Gradient Boosting phù hợp môi trường đang dùng.
- CatBoost, LightGBM hoặc XGBoost nếu thư viện sẵn có và ổn định; với dữ liệu tabular có category, ưu tiên thử CatBoost hoặc boosting có xử lý categorical phù hợp.

Không cần dùng tất cả mô hình nếu chi phí thử nghiệm vượt lợi ích. Chọn 2–4 ứng viên phù hợp với kích thước dữ liệu và môi trường chạy.

### Giai đoạn 4 — Biến đổi target

- Thử train trên `log1p(revenue)` vì target dương, lệch phải mạnh và metric được tính trong không gian log.
- Nếu model dự báo `z = log1p(y)`, chuyển ngược bằng `expm1(z)` và clip kết quả nhỏ hơn 0 về 0.
- So sánh với mô hình train trên target gốc; chỉ chọn theo RMSLE validation, không theo RMSE train.
- Lưu ý: khi mô hình hóa log target, `expm1(predicted_log)` thường phù hợp với mục tiêu log error nhưng không nhất thiết không lệch cho kỳ vọng doanh thu gốc. Không tự ý cộng hệ số hiệu chỉnh nếu chưa đánh giá trên validation.

### Giai đoạn 5 — Lags và ensemble

Chỉ thử sau khi mô hình với calendar, store, promotion, weather và holiday đã được kiểm chứng. Nếu lag features giúp cải thiện, tạo validation recursive giống quy trình test. Có thể thử ensemble trung bình có trọng số giữa hai mô hình tốt nhất nhưng trọng số phải được chọn từ validation và không được dựa trên test label.

### Bảng thí nghiệm tối thiểu

| Experiment | Feature set | Model | Câu hỏi cần trả lời |
|---|---|---|---|
| E0 | Baseline starter | Median store-weekday | Tái lập RMSLE 0.2387 trên validation Q4/2025 (fit đến 30/09/2025) chuẩn theo notebook |
| E1 | Calendar + store | Ridge trên log target | Mùa vụ và metadata có cải thiện baseline không? |
| E2 | E1 + promotion | Ridge/boosting | Khuyến mãi tăng khả năng dự báo đến mức nào? |
| E3 | E2 + holiday/weather | CatBoost/boosting | Các bảng bổ trợ có tăng giá trị out-of-time không? |
| E4 | E3 + lịch sử doanh thu | Boosting với thống kê/lags an toàn | Lags có ích khi mô phỏng dự báo nhiều ngày không? |
| E5 | Top 2 mô hình | Ensemble nếu phù hợp | Ensemble có cải thiện ổn định trên nhiều fold không? |

Mỗi thí nghiệm cần lưu: data version, feature list, cutoff, random seed, hyperparameters, validation RMSLE, metric theo fold/nhóm, thời gian train và đường dẫn artifact.

---

## 9. Tuning, chọn mô hình và phân tích lỗi

### 9.1. Tuning

- Bắt đầu bằng cấu hình mặc định hoặc cấu hình gọn để xác nhận pipeline đúng.
- Dùng random search hoặc Optuna trên các tham số có ảnh hưởng lớn, ví dụ learning rate, depth/num leaves, số cây, regularization, subsampling và l2 regularization.
- Giới hạn tuning bằng validation theo thời gian; dùng early stopping nếu thư viện hỗ trợ.
- Không lặp lại hàng trăm thử nghiệm trên một validation duy nhất nếu không có fold bổ sung, vì sẽ overfit vào validation.

### 9.2. Phân tích residual

Định nghĩa residual log:

\[
r_i = \log(1+y_i)-\log(1+\hat{y}_i)
\]

- `r > 0`: mô hình dự báo thấp; `r < 0`: mô hình dự báo cao.
- Vẽ residual theo ngày, cửa hàng, target thực, `is_promo`, loại cửa hàng và nhiệt độ/mưa.
- Kiểm tra các ngày có lỗi lớn nhất, nhất là dịp lễ, các đợt khuyến mãi và cửa hàng có lịch sử ngắn.
- So sánh dự đoán với baseline trên cùng các dòng để biết mô hình phức tạp đang cải thiện hay chỉ dịch chuyển lỗi.
- Rà soát outliers và data join cho các điểm sai số lớn nhất trước khi thêm features mới.

### 9.3. Tiêu chí chốt mô hình

Mô hình cuối cùng phải:

1. Tốt hơn baseline trong validation chính và có kết quả hợp lý trên fold bổ sung.
2. Không có dấu hiệu leakage; performance không phụ thuộc vào dữ liệu tương lai không khả dụng.
3. Không có lỗi dữ liệu/feature pipeline ở các cửa hàng mới hoặc ngày thiếu.
4. Đạt các kiểm tra kỹ thuật cho inference và xuất kết quả.
5. Được đánh giá bằng RMSLE trên đúng định nghĩa, không chỉ bằng feature importance hay score train.

---

## 10. Huấn luyện cuối cùng và dự báo test

Sau khi cố định mô hình và pipeline:

1. Xác minh file test chính thức có đủ các trường cần thiết (tối thiểu `id`, `date`, `store_id`; `is_promo` nếu được cung cấp).
2. Tạo features train/test bằng cùng một hàm và cùng quy tắc xử lý category, thời tiết, ngày lễ và ngày mở cửa.
3. Fit pipeline đã chọn trên toàn bộ `train.csv` sau kiểm tra chất lượng và deduplication theo quy tắc đã phê duyệt.
4. Dự báo cho từng dòng test. Nếu dùng lag features, tạo theo đúng phương án multi-step đã được đánh giá; không dùng revenue thật ở horizon test.
5. Nếu train trên `log1p(revenue)`, chuyển về thang doanh thu bằng `expm1`.
6. Giới hạn dự đoán không âm; kiểm tra `NaN`, `inf`, giá trị cực đại phi lý và thứ tự dòng.
7. Tạo file submission theo `test.id` đúng thứ tự gốc:

```python
submission = pd.DataFrame({
    "id": test["id"].to_numpy(),
    "revenue": np.clip(predictions, 0, None),
})
submission.to_csv("submission.csv", index=False)
```

Không sắp xếp lại test hoặc submission theo ngày/cửa hàng trừ khi quy định yêu cầu; luôn gắn kết prediction với đúng index của test. Sau khi tạo, kiểm tra:

- `submission.columns == ["id", "revenue"]`;
- số dòng bằng số dòng `test` (theo notebook là 3.869);
- `id` đúng từng vị trí so với test và không bị lặp;
- `revenue` không thiếu, không vô hạn và không âm.

---

## 11. Cấu trúc thư mục đề xuất

```text
cafe-revenue-regression/
├── data/
│   ├── raw/                 # train, test, stores, weather, holidays; không sửa trực tiếp
│   └── processed/           # dữ liệu đã chuẩn hóa, có thể tái tạo
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_quality.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load.py
│   │   ├── validate.py
│   │   └── join.py
│   ├── features/
│   │   ├── calendar.py
│   │   ├── store.py
│   │   ├── weather_holiday.py
│   │   └── historical.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── backtest.py
│   └── utils/
│       └── logging.py
├── tests/
│   ├── test_data_contracts.py
│   ├── test_features.py
│   └── test_submission.py
├── models/                  # model/pipeline đã huấn luyện
├── reports/
│   ├── data_quality_report.md
│   ├── validation_results.csv
│   └── error_analysis.md
├── submissions/
├── requirements.txt
├── README.md
└── Regression_Plan.md
```

Dữ liệu gốc nên được giữ nguyên. Các bước làm sạch, feature engineering và huấn luyện phải chạy lại được từ raw files.

---

## 12. Kế hoạch triển khai theo giai đoạn

| Giai đoạn | Công việc | Đầu ra / tiêu chí hoàn thành |
|---|---|---|
| 0. Chốt đầu vào | Bổ sung `test.csv`, kiểm tra schema, xác nhận metric và forecast horizon | Data contract, mapping ID-test rõ ràng |
| 1. EDA và data audit | Target distribution, độ phủ panel, duplicate, missing, outlier, chuẩn hóa date/category | `data_quality_report.md`, danh sách vấn đề có quyết định xử lý |
| 2. Pipeline dữ liệu | Viết load/validate/join; kiểm soát row count và khóa; tổng hợp holiday | Bảng train/test feature-ready tái lập được |
| 3. Baseline | Tái tạo median store-weekday; tính RMSLE trên time split | Baseline kết quả và file dự đoán validation |
| 4. Feature engineering | Calendar, store metadata, promo, weather/holiday, store age; thử historical features | Bộ features version hóa và kiểm tra leakage |
| 5. Modeling | Ridge/log target, boosting/CatBoost, tuning có kiểm soát | Bảng kết quả các thí nghiệm theo fold |
| 6. Error analysis | Phân tích sai số theo thời gian/cửa hàng/loại ngày, rà soát ngoại lệ | Báo cáo lỗi và quyết định feature/model cuối |
| 7. Final training | Train toàn bộ lịch sử hợp lệ, tạo dự đoán test | Model artifact, feature config và `submission.csv` |
| 8. QA và bàn giao | Kiểm tra cột, số dòng, ID, giá trị đầu ra, phiên bản thư viện | Checklist QA, README hướng dẫn chạy lại |

Có thể thực hiện theo sprint ngắn 5–8 ngày làm việc tùy yêu cầu tuning và mức độ cần điều tra các giá trị bất thường. Nếu chưa có `test.csv` hoặc chưa giải quyết được khóa/ngày lễ, các giai đoạn cuối không được coi là hoàn tất.

---

## 13. Rủi ro chính và phương án giảm thiểu

| Rủi ro | Tác động | Phương án xử lý |
|---|---|---|
| Thiếu `test.csv` | Không thể tạo mapping đúng giữa prediction và `id` | Yêu cầu nguồn test chính thức; không suy luận thứ tự bằng ID nếu không có hợp đồng dữ liệu |
| Trùng bản ghi train | Một số quan sát bị nhân trọng số hoặc làm sai features | Kiểm tra exact/key duplicates, áp dụng quy tắc dedup đã phê duyệt và lưu log |
| Ngoại lệ doanh thu rất lớn | Làm lệch thống kê và đánh giá; có thể là lỗi hoặc sự kiện thật | Điều tra theo cửa hàng/ngày/đơn vị; giữ bản gốc và so sánh cách xử lý theo validation |
| `open_date` có nhiều định dạng | Tuổi cửa hàng bị sai, đặc biệt khi nhầm tháng/ngày | Parse theo pattern cụ thể và kiểm tra với ngày quan sát đầu tiên |
| Holiday coverage chưa đủ | Feature ngày lễ sai trong phần cuối test | Bổ sung lịch đáng tin cậy hoặc đánh dấu coverage; không coi ngày không có record là chắc chắn không lễ |
| Weather missing/outlier | Feature nhiễu và khả năng lệch theo thành phố | Kiểm tra physical range, cờ missing, impute chỉ fit từ history |
| Leakage qua lag/rolling | Validation quá lạc quan so với dự báo thật | Dùng cutoff thời gian và mô phỏng recursive forecast cho horizon nhiều ngày |
| Ít lịch sử ở cửa hàng mới | Dự báo thiếu ổn định theo store ID | Dùng store metadata, weekday/category aggregates có regularization và fallback baseline |
| Overfit vào một quý validation | Điểm số tốt nhưng tổng quát hóa kém | Thêm walk-forward folds, báo cáo kết quả từng fold và phân tích theo nhóm |
| Thời tiết tương lai không có trong production | Hiệu quả notebook không tái hiện khi vận hành | Phân biệt bài toán benchmark và production; dùng weather forecast hoặc bỏ feature không sẵn có |

---

## 14. Checklist trước khi chốt

### Data
- [ ] Đã lấy được `test.csv` chính thức và xác nhận ánh xạ `id`.
- [ ] Các cột ngày đã được parse chính xác; `open_date` được xử lý theo từng định dạng.
- [ ] Đã xử lý và log bản ghi trùng; khóa `(date, store_id)` hợp lệ.
- [ ] Đã điều tra doanh thu cực trị, đặc biệt cụm giá trị của `S12` tháng 03/2024.
- [ ] Đã xử lý `area_m2`, `rain_mm`, nhiệt độ bất thường/thiếu một cách nhất quán.
- [ ] Holiday coverage đến hết horizon đã được xác minh hoặc giới hạn đã được ghi rõ.
- [ ] Sau mỗi join, row count và khóa dòng được kiểm tra.

### Modeling
- [ ] Bảo toàn toàn bộ 26.500 dòng train sạch (sau loại 132 duplicate), không chia file vật lý.
- [ ] Thiết lập Time-based Out-Of-Time Validation (Train: trước 01/10/2025; Val: Q4/2025).
- [ ] Tái lập chuẩn xác baseline RMSLE = 0.2387 trên tập Validation Q4/2025.
- [ ] Chạy Walk-forward Expanding Window Validation để kiểm tra tính ổn định mùa vụ.
- [ ] Feature target-derived không dùng target tương lai (chống rò rỉ dữ liệu).
- [ ] Validation mô phỏng đúng cách dự báo 90 ngày.
- [ ] Metric RMSLE trên thang revenue được dùng để chọn mô hình.
- [ ] Có báo cáo metric tổng thể, theo fold và các nhóm cửa hàng quan trọng.
- [ ] Mô hình cuối tốt hơn baseline (RMSLE < 0.2387) và duy trì ổn định.

### Submission / handoff
- [ ] Nhận file `test.csv` chính thức khi giảng viên upload lên Kaggle.
- [ ] Train lại mô hình đã chọn trên toàn bộ lịch sử 100% sạch (26.500 dòng từ 01/01/2024 đến 31/12/2025).
- [ ] `submission.csv` có đúng hai cột `id`, `revenue` và đúng số dòng test (3.869 dòng).
- [ ] ID giữ nguyên thứ tự chính thức; không thiếu hoặc lặp.
- [ ] Prediction không chứa `NaN`, `inf` hoặc giá trị âm.
- [x] Lưu model, cấu hình, phiên bản thư viện (`requirements.txt`, `venv`), seed, metric và hướng dẫn chạy lại (`README.md`).

---

## 15. Đề xuất ưu tiên thực thi

Theo góc nhìn Senior Data Scientist, nên ưu tiên **độ tin cậy của dữ liệu và validation** trước khi dành nhiều thời gian tuning. Thứ tự đề xuất:

1. Làm sạch dữ liệu, xử lý 132 dòng trùng lặp, chuẩn hóa category/open_date và join an toàn weather/holidays.
2. Thiết lập Time-based OOT Validation (Train: trước 01/10/2025, Val: Q4/2025); tái tạo baseline RMSLE = `0.2387`, sau đó chạy thêm walk-forward expanding window validation.
3. Xây dựng bảng feature theo ngày/cửa hàng, chuẩn hóa cửa hàng, xử lý khoảng cách Tết Âm Lịch 2026.
4. Thử mô hình boosting trên `log1p(revenue)` và so sánh với baseline bằng RMSLE.
5. Bổ sung lag/rolling an toàn (chỉ dùng lịch sử $t-1$) và thử nghiệm Weighted Ensemble.
6. Khi giảng viên upload `test.csv` chính thức lên Kaggle: Fit lại mô hình trên 100% dữ liệu lịch sử sạch (26.500 dòng), dự báo và kiểm tra checklist submission.

**Nguyên tắc quyết định cuối cùng:** chọn pipeline đơn giản nhất cho kết quả out-of-time tốt, ổn định, không leakage và có thể tái lập — không chọn mô hình chỉ vì độ phức tạp hoặc điểm train cao.
