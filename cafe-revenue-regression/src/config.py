from pathlib import Path

# Thư mục gốc của project
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Thư mục chứa dữ liệu gốc
DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"

# Tên các file dữ liệu
TRAIN_FILE = "train.csv"
STORES_FILE = "stores.csv"
WEATHER_FILE = "weather.csv"
HOLIDAYS_FILE = "holidays.csv"
SUBMISSION_FILE = "sample_submission.csv"