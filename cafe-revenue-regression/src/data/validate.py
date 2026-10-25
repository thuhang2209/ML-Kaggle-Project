import logging
from typing import Dict, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def deduplicate_train(df_train: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ các bản ghi trùng khóa store_id + date."""

    required = ["store_id", "date"]
    missing = [col for col in required if col not in df_train.columns]

    if missing:
        raise ValueError(f"Train thiếu các cột: {missing}")

    initial_len = len(df_train)

    df_clean = df_train.drop_duplicates(
        subset=["store_id", "date"],
        keep="first"
    ).copy()

    removed_count = initial_len - len(df_clean)

    logging.info(
        "Deduplication: Removed %s duplicate rows from train data.",
        removed_count
    )

    return df_clean


def validate_schema_and_ranges(
    df_train: pd.DataFrame,
    df_weather: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Kiểm tra schema và các giá trị ngoài phạm vi hợp lý."""

    df_train_clean = df_train.copy()
    df_weather_clean = df_weather.copy()

    # Kiểm tra doanh thu
    if "revenue" in df_train_clean.columns:
        df_train_clean["revenue"] = pd.to_numeric(
            df_train_clean["revenue"],
            errors="coerce"
        )

        invalid_rev = int((df_train_clean["revenue"] < 0).sum())

        if invalid_rev > 0:
            logging.warning(
                "Found %s rows with negative revenue. Clipping to 0.",
                invalid_rev
            )

            df_train_clean["revenue"] = (
                df_train_clean["revenue"].clip(lower=0)
            )

    # Kiểm tra nhiệt độ
    if "temp_c" in df_weather_clean.columns:
        df_weather_clean["temp_c"] = pd.to_numeric(
            df_weather_clean["temp_c"],
            errors="coerce"
        ).clip(lower=-10.0, upper=55.0)

    if "rain_mm" in df_weather_clean.columns:
        df_weather_clean["rain_mm"] = pd.to_numeric(
            df_weather_clean["rain_mm"],
            errors="coerce"
        ).clip(lower=0.0)

    if "temp_max" in df_weather_clean.columns:
        df_weather_clean["temp_max"] = pd.to_numeric(
            df_weather_clean["temp_max"],
            errors="coerce"
        ).clip(lower=-10.0, upper=50.0)

    if "temp_min" in df_weather_clean.columns:
        df_weather_clean["temp_min"] = pd.to_numeric(
            df_weather_clean["temp_min"],
            errors="coerce"
        ).clip(lower=-15.0, upper=45.0)

    # Lượng mưa không âm
    if "precipitation" in df_weather_clean.columns:
        df_weather_clean["precipitation"] = pd.to_numeric(
            df_weather_clean["precipitation"],
            errors="coerce"
        ).clip(lower=0.0)

    # Độ ẩm nằm trong khoảng 0-100%
    if "humidity" in df_weather_clean.columns:
        df_weather_clean["humidity"] = pd.to_numeric(
            df_weather_clean["humidity"],
            errors="coerce"
        ).clip(lower=0.0, upper=100.0)

    # Kiểm tra ngày tháng
    for name, df in [
        ("train", df_train_clean),
        ("weather", df_weather_clean)
    ]:
        if "date" not in df.columns:
            raise ValueError(f"{name} thiếu cột 'date'.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        invalid_dates = int(df["date"].isna().sum())

        if invalid_dates:
            logging.warning(
                "%s contains %s missing or invalid dates.",
                name,
                invalid_dates
            )

    return df_train_clean, df_weather_clean


def handle_missing_values(
    df_weather: pd.DataFrame,
    df_stores: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Xử lý giá trị thiếu trong weather và stores."""

    df_w = df_weather.copy()
    df_s = df_stores.copy()

    # Điền dữ liệu thời tiết theo từng thành phố và thời gian.
    if {"city", "date"}.issubset(df_w.columns):
        df_w = df_w.sort_values(["city", "date"])

        weather_cols = [
            col for col in [
                "temp_c",
                "rain_mm",
                "temp_min",
                "temp_max",
                "precipitation",
                "humidity"
            ]
            if col in df_w.columns
        ]

        for col in weather_cols:
            df_w[col] = df_w.groupby(
                "city",
                observed=True
            )[col].transform(
                lambda group: group.ffill().bfill()
            )

            # Nếu một thành phố thiếu toàn bộ dữ liệu,
            # sử dụng median của toàn bộ bảng.
            if df_w[col].isna().any():
                median_value = df_w[col].median()

                if pd.notna(median_value):
                    df_w[col] = df_w[col].fillna(median_value)

    # Kiểm tra diện tích cửa hàng nếu có.
    if "area_m2" in df_s.columns:
        df_s["area_m2"] = pd.to_numeric(
            df_s["area_m2"],
            errors="coerce"
        )

        median_area = df_s["area_m2"].median()

        if pd.notna(median_area):
            df_s["area_m2"] = df_s["area_m2"].fillna(
                median_area
            )

    logging.info(
        "Missing-value handling completed. Weather shape: %s; stores shape: %s",
        df_w.shape,
        df_s.shape
    )

    return df_w, df_s


def run_validation_pipeline(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Chạy toàn bộ quy trình kiểm tra và làm sạch dữ liệu."""

    logging.info("Starting Data Validation Pipeline...")

    required_keys = {
        "train",
        "stores",
        "weather",
        "holidays",
        "sample_sub"
    }

    missing_keys = required_keys - set(datasets.keys())

    if missing_keys:
        raise KeyError(
            f"Thiếu dataset: {sorted(missing_keys)}"
        )

    validated_datasets = datasets.copy()

    # Loại bỏ các dòng train trùng khóa.
    train_clean = deduplicate_train(datasets["train"])

    # Kiểm tra phạm vi dữ liệu.
    train_clean, weather_clean = validate_schema_and_ranges(
        train_clean,
        datasets["weather"]
    )

    # Xử lý dữ liệu thiếu.
    weather_clean, stores_clean = handle_missing_values(
        weather_clean,
        datasets["stores"]
    )

    validated_datasets["train"] = train_clean
    validated_datasets["weather"] = weather_clean
    validated_datasets["stores"] = stores_clean

    logging.info("Data Validation Pipeline Completed Successfully.")

    return validated_datasets


if __name__ == "__main__":
    from src.data.load import load_raw_data

    raw_data = load_raw_data()
    clean_data = run_validation_pipeline(raw_data)

    print("\n--- Validation Summary ---")

    for name, df in clean_data.items():
        print(f"{name:12s} -> Shape: {df.shape}")

