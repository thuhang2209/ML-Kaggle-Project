from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    STORES_FILE,
    TRAIN_FILE,
    WEATHER_FILE,
    HOLIDAYS_FILE,
    SUBMISSION_FILE,
)


def load_raw_data(
    data_dir: Path = DATA_RAW_DIR,
) -> Dict[str, pd.DataFrame]:

    data_dir = Path(data_dir)

    # 1. Train
    train_path = data_dir / TRAIN_FILE
    df_train = pd.read_csv(train_path)

    df_train["date"] = pd.to_datetime(
        df_train["date"],
        errors="coerce",
    )

    if "store_id" in df_train.columns:
        df_train["store_id"] = df_train["store_id"].astype("category")

    if "revenue" in df_train.columns:
        df_train["revenue"] = pd.to_numeric(
            df_train["revenue"],
            errors="coerce",
        ).astype("float32")

    # 2. Stores
    stores_path = data_dir / STORES_FILE
    df_stores = pd.read_csv(stores_path)

    if "store_id" in df_stores.columns:
        df_stores["store_id"] = df_stores["store_id"].astype("category")

    if "city" in df_stores.columns:
        df_stores["city"] = (
            df_stores["city"].astype("string").str.strip()
        )

    # Chuẩn hóa open_date hỗ trợ cả ISO (YYYY-MM-DD) và DD/MM/YYYY
    if "open_date" in df_stores.columns:
        s = df_stores["open_date"].astype("string").str.strip()
        parsed_dates = pd.Series(pd.NaT, index=df_stores.index, dtype="datetime64[ns]")
        iso_mask = s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$", na=False)
        parsed_dates.loc[iso_mask] = pd.to_datetime(s.loc[iso_mask], format="%Y-%m-%d", errors="coerce")
        parsed_dates.loc[~iso_mask] = pd.to_datetime(s.loc[~iso_mask], format="%d/%m/%Y", errors="coerce")
        df_stores["open_date"] = parsed_dates

    if "opening_date" in df_stores.columns:
        df_stores["opening_date"] = pd.to_datetime(
            df_stores["opening_date"],
            errors="coerce",
        )

    if "area_m2" in df_stores.columns:
        df_stores["area_m2"] = pd.to_numeric(
            df_stores["area_m2"],
            errors="coerce",
        ).astype("float32")

    if "seats" in df_stores.columns:
        df_stores["seats"] = pd.to_numeric(
            df_stores["seats"],
            errors="coerce",
        ).astype("Int32")

    if "capacity" in df_stores.columns:
        df_stores["capacity"] = pd.to_numeric(
            df_stores["capacity"],
            errors="coerce",
        ).astype("Int32")

    # 3. Weather
    weather_path = data_dir / WEATHER_FILE
    df_weather = pd.read_csv(weather_path)

    df_weather["date"] = pd.to_datetime(
        df_weather["date"],
        errors="coerce",
    )

    if "city" in df_weather.columns:
        df_weather["city"] = (
            df_weather["city"].astype("string").str.strip()
        )

    for col in ["temp_c", "rain_mm", "temp_min", "temp_max", "precipitation", "humidity"]:
        if col in df_weather.columns:
            df_weather[col] = pd.to_numeric(
                df_weather[col],
                errors="coerce",
            ).astype("float32")

    # 4. Holidays
    holidays_path = data_dir / HOLIDAYS_FILE
    df_holidays = pd.read_csv(holidays_path)

    df_holidays["date"] = pd.to_datetime(
        df_holidays["date"],
        errors="coerce",
    )

    for col in ["holiday_name", "type", "holiday_type"]:
        if col in df_holidays.columns:
            df_holidays[col] = (
                df_holidays[col].astype("string").str.strip()
            )

    # 5. Sample submission
    sub_path = data_dir / SUBMISSION_FILE
    df_sub = pd.read_csv(sub_path)

    if "date" in df_sub.columns:
        df_sub["date"] = pd.to_datetime(
            df_sub["date"],
            errors="coerce",
        )

    if "store_id" in df_sub.columns:
        df_sub["store_id"] = df_sub["store_id"].astype("category")

    # 6. Kiểm tra file và dữ liệu
    datasets = {
        "train": df_train,
        "stores": df_stores,
        "weather": df_weather,
        "holidays": df_holidays,
        "sample_sub": df_sub,
    }

    for name, df in datasets.items():
        if df.empty:
            raise ValueError(
                f"Bộ dữ liệu '{name}' đang rỗng."
            )

        if "date" in df.columns:
            invalid_dates = int(df["date"].isna().sum())

            if invalid_dates:
                print(
                    f"Cảnh báo {name}: {invalid_dates} "
                    "giá trị date bị thiếu hoặc không hợp lệ."
                )

        print(
            f"{name:12s} -> Shape: {df.shape}"
        )

    return datasets


if __name__ == "__main__":
    datasets = load_raw_data()

    print("\n--- Loaded Datasets Summary ---")

    for name, df in datasets.items():
        print(f"\n{name.upper()}")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head(3).to_string(index=False))
