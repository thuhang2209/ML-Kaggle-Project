

from pathlib import Path
import pandas as pd


# ==================================================
# 1. XÁC ĐỊNH THƯ MỤC PROJECT
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

REQUIRED_FILES = {
    "train": "train.csv",
    "stores": "stores.csv",
    "weather": "weather.csv",
    "holidays": "holidays.csv",
}


# ==================================================
# 2. TÌM THƯ MỤC CHỨA DỮ LIỆU
# ==================================================

def find_data_dir():
    candidates = [
        PROJECT_DIR / "data" / "raw",
        PROJECT_DIR / "data",
        PROJECT_DIR / "cafe" / "data" / "raw",
        PROJECT_DIR / "cafe" / "data",
        PROJECT_DIR,
    ]

    # Tìm thêm trong các thư mục con
    if PROJECT_DIR.exists():
        candidates.extend(
            p for p in PROJECT_DIR.rglob("*")
            if p.is_dir()
            and "node_modules" not in p.parts
            and ".venv" not in p.parts
            and ".git" not in p.parts
        )

    checked = set()

    for folder in candidates:
        folder = folder.resolve()

        if folder in checked:
            continue

        checked.add(folder)

        if all(
            (folder / filename).is_file()
            for filename in REQUIRED_FILES.values()
        ):
            print("Đã tìm thấy dữ liệu:", folder)
            return folder

    # Nếu không đủ 4 file, báo cụ thể
    print("Project đang tìm dữ liệu tại:", PROJECT_DIR)
    print("\nCác file CSV tìm thấy:")

    for filename in REQUIRED_FILES.values():
        matches = list(PROJECT_DIR.rglob(filename))
        if matches:
            for path in matches:
                print(f"  {filename}: {path}")
        else:
            print(f"  {filename}: KHÔNG TÌM THẤY")

    raise FileNotFoundError(
        "Không tìm thấy thư mục chứa đủ 4 file CSV. "
        "Kiểm tra dữ liệu đã được giải nén đầy đủ chưa."
    )


# ==================================================
# 3. CHUẨN HÓA ĐỊNH DẠNG NGÀY
# ==================================================

def parse_mixed_dates(series):
    values = series.astype("string").str.strip()

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    # YYYY-MM-DD
    iso_mask = values.str.fullmatch(
        r"\d{4}-\d{1,2}-\d{1,2}",
        na=False
    )

    # DD/MM/YYYY
    dmy_mask = values.str.fullmatch(
        r"\d{1,2}/\d{1,2}/\d{4}",
        na=False
    )

    result.loc[iso_mask] = pd.to_datetime(
        values.loc[iso_mask],
        format="%Y-%m-%d",
        errors="coerce"
    )

    result.loc[dmy_mask] = pd.to_datetime(
        values.loc[dmy_mask],
        format="%d/%m/%Y",
        errors="coerce"
    )

    return result


# ==================================================
# 4. ĐỌC TOÀN BỘ DỮ LIỆU
# ==================================================

def load_all(data_dir=None):
    if data_dir is None:
        data_dir = find_data_dir()
    else:
        data_dir = Path(data_dir).resolve()

    data = {}

    for name, filename in REQUIRED_FILES.items():
        file_path = data_dir / filename

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy file: {file_path}"
            )

        df = pd.read_csv(file_path)

        # Chuẩn hóa cột ngày nếu tồn tại
        if "date" in df.columns:
            original_dates = df["date"].copy()
            df["date"] = parse_mixed_dates(original_dates)

            invalid_count = int(df["date"].isna().sum())

            if invalid_count:
                print(
                    f"Cảnh báo {name}: {invalid_count} "
                    "giá trị ngày bị thiếu hoặc không hợp lệ."
                )

        data[name] = df

    return data


# ==================================================
# 5. CHẠY KIỂM TRA
# ==================================================

if __name__ == "__main__":
    datasets = load_all()

    print("\n========== KẾT QUẢ ĐỌC DỮ LIỆU ==========")

    for name, df in datasets.items():
        print(f"\n{name.upper()}")
        print("Kích thước:", df.shape)
        print("Các cột:", df.columns.tolist())
        print(df.head(3).to_string(index=False))

    print("\nĐã đọc xong dữ liệu.")
