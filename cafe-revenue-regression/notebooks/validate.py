
from pathlib import Path
import sys
import pandas as pd

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = NOTEBOOK_DIR.parent

sys.path.insert(0, str(NOTEBOOK_DIR))

from load import load_all


def main():
    print("=" * 60)
    print("DATA QUALITY VALIDATION")
    print("=" * 60)

    result = load_all()

    # Nhận diện dữ liệu trả về từ load_all()
    if isinstance(result, dict):
        datasets = result
    elif isinstance(result, (tuple, list)):
        if len(result) != 4:
            raise ValueError(
                f"load_all() phải trả về 4 bộ dữ liệu, "
                f"nhưng nhận được {len(result)} giá trị."
            )

        datasets = dict(zip(
            ["train", "stores", "weather", "holidays"],
            result
        ))
    else:
        raise TypeError(
            "load_all() phải trả về dictionary, tuple hoặc list."
        )

    # Kiểm tra kiểu dữ liệu
    for name, df in datasets.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"'{name}' có kiểu {type(df).__name__}, "
                "không phải DataFrame. "
                "Hãy kiểm tra giá trị trả về trong load.py."
            )

    print("\nKÍCH THƯỚC DỮ LIỆU")
    for name, df in datasets.items():
        print(f"{name}: {df.shape[0]} dòng, {df.shape[1]} cột")

    errors = []
    warnings = []

    required_columns = {
        "train": ["date", "store_id", "is_promo", "revenue"],
        "stores": ["store_id"],
        "weather": ["date"],
        "holidays": ["date"],
    }

    for name, columns in required_columns.items():
        missing = [
            col for col in columns
            if col not in datasets[name].columns
        ]

        if missing:
            errors.append(f"{name}: thiếu cột {missing}")

    if errors:
        print("\nLỖI:")
        for error in errors:
            print("-", error)
        sys.exit(1)

    train = datasets["train"]
    stores = datasets["stores"]
    weather = datasets["weather"]

    # Kiểm tra giá trị thiếu
    print("\nGIÁ TRỊ THIẾU")
    for name, df in datasets.items():
        missing = int(df.isna().sum().sum())
        print(f"{name}: {missing} ô thiếu")

        if missing:
            warnings.append(
                f"{name}: có {missing} ô dữ liệu thiếu"
            )

    # Kiểm tra dòng trùng
    print("\nDÒNG TRÙNG LẶP")
    for name, df in datasets.items():
        count = int(df.duplicated().sum())
        print(f"{name}: {count} dòng trùng hoàn toàn")

        if count:
            warnings.append(
                f"{name}: có {count} dòng trùng hoàn toàn"
            )

    # Kiểm tra khóa date + store_id
    duplicate_keys = int(
        train.duplicated(subset=["date", "store_id"]).sum()
    )
    print(f"\nTrùng khóa date + store_id: {duplicate_keys}")

    if duplicate_keys:
        warnings.append(
            f"train: có {duplicate_keys} dòng trùng khóa date + store_id"
        )

    # Kiểm tra doanh thu
    revenue = pd.to_numeric(train["revenue"], errors="coerce")

    negative = int((revenue < 0).sum())
    zero = int((revenue == 0).sum())
    outliers = int((revenue > 1000).sum())

    print("\nKIỂM TRA DOANH THU")
    print(f"Doanh thu âm: {negative}")
    print(f"Doanh thu bằng 0: {zero}")
    print(f"Doanh thu > 1000: {outliers}")

    if revenue.isna().sum() > train["revenue"].isna().sum():
        warnings.append("revenue có giá trị không phải số")

    if negative:
        warnings.append(f"Có {negative} doanh thu âm")

    if zero:
        warnings.append(f"Có {zero} doanh thu bằng 0")

    if outliers:
        warnings.append(
            f"Có {outliers} doanh thu > 1000, cần kiểm tra"
        )

    # Kiểm tra store_id không tồn tại trong stores
    unknown_stores = (
        set(train["store_id"].dropna().unique())
        - set(stores["store_id"].dropna().unique())
    )

    if unknown_stores:
        warnings.append(
            f"store_id không có trong stores.csv: "
            f"{sorted(unknown_stores)}"
        )

    # Kiểm tra diện tích cửa hàng
    if "area_m2" in stores.columns:
        area = pd.to_numeric(stores["area_m2"], errors="coerce")

        print("\nKIỂM TRA DIỆN TÍCH")
        print(f"area_m2 thiếu/không hợp lệ: {int(area.isna().sum())}")
        print(f"area_m2 âm: {int((area < 0).sum())}")

    # Kiểm tra lượng mưa
    if "rain_mm" in weather.columns:
        rain = pd.to_numeric(weather["rain_mm"], errors="coerce")

        print("\nKIỂM TRA LƯỢNG MƯA")
        print(f"rain_mm thiếu/không hợp lệ: {int(rain.isna().sum())}")
        print(f"rain_mm âm: {int((rain < 0).sum())}")

    print("\nCẢNH BÁO")
    if warnings:
        for warning in warnings:
            print("-", warning)
    else:
        print("Không có cảnh báo.")

    report_path = PROJECT_DIR / "validation_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("DATA QUALITY VALIDATION REPORT\n")
        file.write("=" * 40 + "\n\n")

        for name, df in datasets.items():
            file.write(
                f"{name}: {df.shape[0]} dòng, {df.shape[1]} cột\n"
            )

        file.write("\nERRORS\n")
        file.write("\n".join(f"- {x}" for x in errors) or "Không có lỗi.")

        file.write("\n\nWARNINGS\n")
        file.write(
            "\n".join(f"- {x}" for x in warnings)
            or "Không có cảnh báo."
        )

    print(f"\nĐã lưu báo cáo: {report_path}")


if __name__ == "__main__":
    main()


