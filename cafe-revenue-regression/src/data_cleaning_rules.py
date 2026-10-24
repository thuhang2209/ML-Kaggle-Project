import pandas as pd
import numpy as np

def parse_mixed_dates(values):
    s = values.astype("string").str.strip()
    iso = s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$", na=False)

    result = pd.Series(
        pd.NaT, index=values.index, dtype="datetime64[ns]"
    )

    result.loc[iso] = pd.to_datetime(
        s.loc[iso], format="%Y-%m-%d", errors="coerce"
    )
    result.loc[~iso] = pd.to_datetime(
        s.loc[~iso], dayfirst=True, errors="coerce"
    )
    return result


def clean_stores(stores):
    df = stores.copy()

    if "open_date" in df.columns:
        df["open_date"] = parse_mixed_dates(df["open_date"])

    if "area_m2" in df.columns:
        df["area_m2"] = pd.to_numeric(
            df["area_m2"], errors="coerce"
        )

    for col in ["city", "district", "store_type"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    return df


def clean_weather(weather):
    df = weather.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )

    for col in ["temp_c", "rain_mm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    return df


def clean_holidays(holidays):
    df = holidays.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )

    return df


def clean_train(train):
    df = train.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )

    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(
            df["revenue"], errors="coerce"
        )
        df["revenue_extreme_flag"] = df["revenue"] > 1000

    return df
