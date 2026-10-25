# BÁO CÁO CHẤT LƯỢNG DỮ LIỆU — DỰ ĐOÁN DOANH THU QUÁN CÀ PHÊ

## 1. Mục tiêu

Báo cáo nhằm đánh giá chất lượng dữ liệu trước khi xây dựng mô hình Machine Learning dự đoán doanh thu quán cà phê. Các nội dung chính bao gồm kiểm tra giá trị thiếu, dữ liệu trùng lặp, giá trị ngoại lệ, kiểu dữ liệu và tính nhất quán giữa các bảng dữ liệu.

## 2. Giới thiệu các bộ dữ liệu

Dự án sử dụng bốn bộ dữ liệu chính:

| Tên bộ dữ liệu | Nội dung |
|---|---|
| `train.csv` | Doanh thu theo ngày, mã cửa hàng và trạng thái khuyến mãi |
| `stores.csv` | Thông tin các cửa hàng |
| `weather.csv` | Thông tin thời tiết |
| `holidays.csv` | Thông tin ngày lễ |

## 3. Đánh giá chất lượng dữ liệu

### 3.1. Kiểm tra giá trị thiếu

Tiến hành kiểm tra số lượng và tỷ lệ giá trị thiếu trong từng cột của các bộ dữ liệu.

Giá trị thiếu có thể ảnh hưởng đến quá trình phân tích và huấn luyện mô hình. Vì vậy, cần xác định nguyên nhân và ý nghĩa của các giá trị thiếu trước khi xử lý.

**Hướng xử lý đề xuất:**
- Xác định các cột có giá trị thiếu.
- Kiểm tra nguyên nhân thiếu dữ liệu.
- Lựa chọn phương pháp xử lý phù hợp với từng cột.
- Không tự động thay thế toàn bộ giá trị thiếu bằng 0 khi chưa xác định ý nghĩa của biến.

### 3.2. Kiểm tra dữ liệu trùng lặp

Kiểm tra hai dạng dữ liệu trùng lặp:

- Các dòng có toàn bộ giá trị giống nhau.
- Các bản ghi có cùng `date` và `store_id` trong `train.csv`.

Các bản ghi trùng khóa cần được kiểm tra trước khi quyết định loại bỏ. Việc trùng khóa có thể xuất phát từ lỗi thu thập dữ liệu hoặc cách ghi nhận doanh thu.

### 3.3. Kiểm tra giá trị doanh thu ngoại lệ

Sử dụng thống kê mô tả và phương pháp khoảng tứ phân vị (IQR) để xác định các giá trị doanh thu có khả năng bất thường.

Các giá trị doanh thu âm, bằng 0 hoặc lớn bất thường cũng cần được kiểm tra.

Tuy nhiên, giá trị ngoại lệ không nhất thiết là dữ liệu sai. Doanh thu cao có thể xuất hiện vào các ngày lễ, dịp khuyến mãi hoặc thời điểm kinh doanh đặc biệt.

**Hướng xử lý đề xuất:**
- Kiểm tra các giá trị ngoại lệ với dữ liệu gốc.
- Đối chiếu với ngày tháng, cửa hàng và chương trình khuyến mãi.
- Chỉ loại bỏ hoặc biến đổi dữ liệu khi có căn cứ rõ ràng.
- Không tự động xóa các giá trị doanh thu lớn chỉ vì chúng nằm ngoài ngưỡng IQR.

### 3.4. Kiểm tra kiểu dữ liệu và ngày tháng

Kiểm tra kiểu dữ liệu của các cột và tính hợp lệ của các trường ngày tháng.

Các cột ngày tháng cần được chuyển về định dạng thống nhất để phục vụ việc kết hợp dữ liệu và phân tích xu hướng doanh thu.

Các cột số cần được chuyển sang kiểu dữ liệu phù hợp. Những giá trị không thể chuyển đổi cần được kiểm tra và xử lý trước khi xây dựng mô hình.

### 3.5. Kiểm tra tính nhất quán giữa các bảng

Kiểm tra sự tương thích giữa các bộ dữ liệu, đặc biệt là mã cửa hàng và ngày tháng.

Các nội dung cần kiểm tra gồm:
- Mã cửa hàng trong `train.csv` có tồn tại trong `stores.csv` hay không.
- Các trường ngày tháng có cùng định dạng và phạm vi thời gian phù hợp hay không.
- Giá trị diện tích cửa hàng `area_m2` có bị thiếu hoặc âm hay không, nếu cột này tồn tại.
- Giá trị lượng mưa `rain_mm` có bị thiếu hoặc âm hay không, nếu cột này tồn tại.
- Các khóa liên kết có phù hợp trước khi kết hợp dữ liệu hay không.

## 4. Quy tắc làm sạch dữ liệu đề xuất

Dựa trên các nhóm kiểm tra chất lượng dữ liệu, đề xuất những quy tắc sau:

1. Chuẩn hóa định dạng ngày tháng giữa các bộ dữ liệu.
2. Chuyển đổi kiểu dữ liệu của các cột số và cột phân loại cho phù hợp.
3. Kiểm tra dữ liệu trùng lặp trước khi quyết định loại bỏ.
4. Xử lý giá trị thiếu dựa trên ý nghĩa và đặc điểm của từng biến.
5. Kiểm tra các giá trị doanh thu âm, bằng 0 và ngoại lệ trước khi quyết định cách xử lý.
6. Kiểm tra tính hợp lệ của mã cửa hàng trước khi kết hợp các bảng.
7. Giữ nguyên dữ liệu gốc và thực hiện làm sạch trên bản sao để bảo đảm khả năng tái lập.
8. Tránh rò rỉ dữ liệu (data leakage) bằng cách chỉ sử dụng thông tin có sẵn tại thời điểm dự đoán.

## 5. Ảnh hưởng đến mô hình Machine Learning

Các vấn đề về chất lượng dữ liệu có thể ảnh hưởng trực tiếp đến kết quả dự đoán:

- **Giá trị thiếu:** Có thể khiến một số thuật toán không huấn luyện được hoặc làm giảm chất lượng mô hình.
- **Dữ liệu trùng lặp:** Có thể khiến một số quan sát được tính nhiều lần, làm sai lệch phân phối dữ liệu.
- **Giá trị ngoại lệ:** Có thể ảnh hưởng đáng kể đến các mô hình nhạy cảm với giá trị lớn.
- **Kiểu dữ liệu không thống nhất:** Có thể gây lỗi khi xử lý dữ liệu hoặc kết hợp các bảng.
- **Mã cửa hàng không nhất quán:** Có thể làm mất thông tin khi kết hợp dữ liệu.
- **Rò rỉ dữ liệu:** Có thể khiến kết quả đánh giá mô hình cao hơn khả năng dự đoán thực tế.

Do đó, cần kiểm tra và xử lý các vấn đề trên trước khi xây dựng pipeline huấn luyện và đánh giá mô hình.

## 6. Tổng hợp kết quả kiểm tra

Điền số liệu thực tế sau khi chạy notebook `02_data_quality.ipynb`.

| Chỉ tiêu | Kết quả |
|---|---|
| Tổng số dòng của `train.csv` | Cập nhật từ kết quả chạy |
| Tổng số giá trị thiếu trong `train.csv` | Cập nhật từ kết quả chạy |
| Số dòng trùng lặp hoàn toàn trong `train.csv` | Cập nhật từ kết quả chạy |
| Số dòng thuộc nhóm trùng khóa `date` và `store_id` | Cập nhật từ kết quả chạy |
| Số giá trị doanh thu ngoại lệ theo IQR | Cập nhật từ kết quả chạy |
| Số giá trị doanh thu âm | Cập nhật từ kết quả chạy |
| Số giá trị doanh thu bằng 0 | Cập nhật từ kết quả chạy |
| Số mã cửa hàng không tồn tại trong `stores.csv` | Cập nhật từ kết quả chạy |

## 7. Kết luận

Việc đánh giá chất lượng dữ liệu giúp xác định những vấn đề có thể ảnh hưởng đến quá trình phân tích và dự đoán doanh thu quán cà phê. Các nhóm kiểm tra chính bao gồm giá trị thiếu, dữ liệu trùng lặp, giá trị ngoại lệ, kiểu dữ liệu và tính nhất quán giữa các bảng.

Các quy tắc làm sạch cần được áp dụng dựa trên kết quả kiểm tra thực tế và ý nghĩa nghiệp vụ của từng biến. Dữ liệu gốc cần được bảo toàn, đồng thời mọi thay đổi trong quá trình làm sạch cần được ghi nhận để bảo đảm tính minh bạch và khả năng tái lập.

Sau khi hoàn thành việc kiểm tra và làm sạch dữ liệu, bộ dữ liệu có thể được sử dụng cho các bước xây dựng đặc trưng, huấn luyện và đánh giá mô hình dự đoán doanh thu.
