import logging
from typing import Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def aggregate_holidays(
    df_holidays: pd.DataFrame
) -> pd.DataFrame:
    """Gom nhóm các ngày lễ theo ngày."""

    holidays = df_holidays.copy()

    if "date" not in holidays.columns:
        raise ValueError("Bảng holidays thiếu cột 'date'.")

    if "holiday_name" in holidays.columns:
        name_col = "holiday_name"
    elif "type" in holidays.columns:
        name_col = "type"
    else:
        raise ValueError(
            "Bảng holidays cần có cột 'holiday_name' hoặc 'type'."
        )

    holiday_agg = (
        holidays.groupby("date", as_index=False)
        .agg(
            is_holiday=(name_col, "count"),
            holiday_names=(
                name_col,
                lambda values: "|".join(
                    values.dropna().astype(str).unique()
                )
            )
        )
    )

    holiday_agg["is_holiday"] = (
        holiday_agg["is_holiday"] > 0
    ).astype("int8")

    return holiday_agg


def merge_datasets(
    main_df: pd.DataFrame,
    df_stores: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_holidays: pd.DataFrame
) -> pd.DataFrame:
    """Join dữ liệu doanh thu với cửa hàng, thời tiết và ngày lễ."""

    required_main = {"date", "store_id"}
    required_stores = {"store_id", "city"}
    required_weather = {"date", "city", "temp_c", "rain_mm"}

    if not required_main.issubset(main_df.columns):
        raise ValueError(
            f"Dữ liệu chính thiếu cột: "
            f"{required_main - set(main_df.columns)}"
        )

    if not required_stores.issubset(df_stores.columns):
        raise ValueError(
            f"Stores thiếu cột: "
            f"{required_stores - set(df_stores.columns)}"
        )

    if not required_weather.issubset(df_weather.columns):
        raise ValueError(
            f"Weather thiếu cột: "
            f"{required_weather - set(df_weather.columns)}"
        )

    initial_rows = len(main_df)

    # Chuẩn hóa kiểu dữ liệu của khóa join.
    main = main_df.copy()
    stores = df_stores.copy()
    weather = df_weather.copy()
    holidays = df_holidays.copy()

    main["date"] = pd.to_datetime(main["date"])
    weather["date"] = pd.to_datetime(weather["date"])
    holidays["date"] = pd.to_datetime(holidays["date"])

    # 1. Join thông tin cửa hàng.
    merged = main.merge(
        stores,
        on="store_id",
        how="left",
        validate="many_to_one",
        indicator="_store_match"
    )

    if (merged["_store_match"] == "left_only").any():
        missing_ids = (
            merged.loc[
                merged["_store_match"] == "left_only",
                "store_id"
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Không tìm thấy thông tin cho các store_id: {missing_ids}"
        )

    merged = merged.drop(columns="_store_match")

    assert len(merged) == initial_rows

    # 2. Join thông tin thời tiết.
    merged = merged.merge(
        weather,
        on=["date", "city"],
        how="left",
        validate="many_to_one"
    )

    assert len(merged) == initial_rows, (
        "Join weather làm thay đổi số dòng."
    )

    # 3. Join thông tin ngày lễ.
    holiday_agg = aggregate_holidays(holidays)

    merged = merged.merge(
        holiday_agg,
        on="date",
        how="left",
        validate="many_to_one"
    )

    assert len(merged) == initial_rows, (
        "Join holidays làm thay đổi số dòng."
    )

    merged["is_holiday"] = (
        merged["is_holiday"]
        .fillna(0)
        .astype("int8")
    )

    merged["holiday_names"] = (
        merged["holiday_names"].fillna("")
    )

    logging.info(
        "Join successful: %s rows, %s columns",
        len(merged),
        len(merged.columns)
    )

    return merged


def build_feature_ready_data(
    validated_datasets: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df_train = validated_datasets["train"]
    df_stores = validated_datasets["stores"]
    df_weather = validated_datasets["weather"]
    df_holidays = validated_datasets["holidays"]

    processed_train = merge_datasets(
        df_train,
        df_stores,
        df_weather,
        df_holidays
    )

    # Chưa thể tạo test data khi chưa có date và store_id.
    logging.warning(
        "sample_sub chỉ có id và revenue. "
        "Cần xác định quy tắc id trước khi tạo test data."
    )

    return processed_train, validated_datasets["sample_sub"]


if __name__ == "__main__":
    from src.data.load import load_raw_data
    from src.data.validate import run_validation_pipeline

    raw_data = load_raw_data()
    clean_data = run_validation_pipeline(raw_data)

    train_ready, sample_sub = build_feature_ready_data(clean_data)

    print("Train ready shape:", train_ready.shape)
    print("Sample submission shape:", sample_sub.shape)
    print(train_ready.head())

