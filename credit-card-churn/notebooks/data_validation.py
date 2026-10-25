
import argparse
from pathlib import Path
import pandas as pd


def find_column(df, candidates):
    columns = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for name in candidates:
        key = name.strip().lower().replace(" ", "_")
        if key in columns:
            return columns[key]

    return None


def find_input_file(project_dir):
    preferred_names = [
        "churn.csv",
        "credit_card_churn.csv",
        "creditcard_churn.csv",
        "train.csv",
        "data.csv"
    ]

    csv_files = [
        path for path in project_dir.rglob("*.csv")
        if not any(
            word in path.stem.lower()
            for word in ["validated", "report", "cleaned"]
        )
    ]

    for name in preferred_names:
        matches = [
            path for path in csv_files
            if path.name.lower() == name.lower()
        ]

        if matches:
            return matches[0]

    if csv_files:
        return max(csv_files, key=lambda path: path.stat().st_size)

    raise FileNotFoundError(
        f"Không tìm thấy file CSV trong thư mục: {project_dir}\n"
        "Hãy kiểm tra dữ liệu đã được giải nén chưa."
    )


def validate_data(df, age_min=18, age_max=100, income_min=0):
    df = df.copy()

    rows_before = len(df)

    # 1. Xóa các dòng trùng hoàn toàn
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    print("\n--- KIỂM TRA DÒNG TRÙNG ---")
    print("Số dòng ban đầu:", rows_before)
    print("Số dòng trùng đã xóa:", duplicate_count)
    print("Số dòng sau khi xóa trùng:", len(df))

    # 2. Kiểm tra tuổi
    age_col = find_column(df, ["age", "customer_age"])

    print("\n--- KIỂM TRA TUỔI ---")

    if age_col is not None:
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

        age_out_of_range = (
            df[age_col].notna()
            & (
                (df[age_col] < age_min)
                | (df[age_col] > age_max)
            )
        )

        age_missing = df[age_col].isna()

        df["age_invalid"] = age_missing | age_out_of_range

        print("Cột tuổi:", age_col)
        print("Giá trị tuổi thiếu/không phải số:", int(age_missing.sum()))
        print("Tuổi ngoài phạm vi:", int(age_out_of_range.sum()))
        print(f"Phạm vi tuổi hợp lệ: {age_min} - {age_max}")
    else:
        print("Không tìm thấy cột age hoặc customer_age.")

    # 3. Kiểm tra thu nhập
    income_col = find_column(
        df,
        [
            "income",
            "income_month",
            "monthly_income",
            "estimated_salary",
            "salary"
        ]
    )

    print("\n--- KIỂM TRA THU NHẬP ---")

    if income_col is not None:
        df[income_col] = pd.to_numeric(
            df[income_col], errors="coerce"
        )

        income_out_of_range = (
            df[income_col].notna()
            & (df[income_col] < income_min)
        )

        income_missing = df[income_col].isna()

        df["income_invalid"] = (
            income_missing | income_out_of_range
        )

        print("Cột thu nhập:", income_col)
        print(
            "Giá trị thu nhập thiếu/không phải số:",
            int(income_missing.sum())
        )
        print(
            "Thu nhập nhỏ hơn ngưỡng:",
            int(income_out_of_range.sum())
        )
        print("Ngưỡng thu nhập tối thiểu:", income_min)
    else:
        print("Không tìm thấy cột income hoặc salary.")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra dữ liệu Credit Card Churn"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Đường dẫn CSV đầu vào"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn CSV đầu ra"
    )

    parser.add_argument("--age-min", type=float, default=18)
    parser.add_argument("--age-max", type=float, default=100)
    parser.add_argument("--income-min", type=float, default=0)

    args = parser.parse_args()

    # File hiện tại ở notebooks/, nên thư mục dự án là thư mục cha
    project_dir = Path(__file__).resolve().parent.parent

    # Tìm file dữ liệu
    if args.input:
        input_path = Path(args.input).expanduser().resolve()

        if not input_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy file đầu vào: {input_path}"
            )
    else:
        input_path = find_input_file(project_dir)

    print("=" * 55)
    print("DATA VALIDATION - CREDIT CARD CHURN")
    print("=" * 55)
    print("Thư mục dự án:", project_dir)
    print("File đầu vào:", input_path)

    # Đọc dữ liệu
    df = pd.read_csv(input_path)

    print("Kích thước dữ liệu ban đầu:", df.shape)

    # Kiểm tra và làm sạch
    validated_df = validate_data(
        df,
        age_min=args.age_min,
        age_max=args.age_max,
        income_min=args.income_min
    )

    # Lưu dữ liệu đã kiểm tra
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_name(
            f"{input_path.stem}_validated.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    validated_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Tạo báo cáo
    report_path = output_path.with_name(
        "data_validation_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("BÁO CÁO KIỂM TRA DỮ LIỆU - CREDIT CARD CHURN\n")
        file.write("=" * 55 + "\n")
        file.write(f"File đầu vào: {input_path}\n")
        file.write(f"Số dòng ban đầu: {len(df)}\n")
        file.write(
            f"Số dòng trùng đã xóa: {int(df.duplicated().sum())}\n"
        )
        file.write(f"Số dòng sau kiểm tra: {len(validated_df)}\n")

        for col in ["age_invalid", "income_invalid"]:
            if col in validated_df.columns:
                file.write(
                    f"Số dòng bị đánh dấu {col}: "
                    f"{int(validated_df[col].sum())}\n"
                )

    print("\n--- KẾT QUẢ ---")
    print("Kích thước sau kiểm tra:", validated_df.shape)
    print("File dữ liệu đầu ra:", output_path)
    print("File báo cáo:", report_path)

    actual_duplicates = int(df.duplicated().sum())

    if actual_duplicates != 120:
        print(
            "\nLƯU Ý: Đề bài yêu cầu loại bỏ 120 dòng trùng, "
            f"nhưng file hiện tại có {actual_duplicates} dòng trùng hoàn toàn."
        )
        print(
            "Hãy xác nhận đúng file dữ liệu và quy tắc xác định dòng trùng."
        )


if __name__ == "__main__":
    main()

