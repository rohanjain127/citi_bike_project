import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import requests

from src.config import RAW_DATA_DIR

# ----------------------------------
# 1. Fetch raw Citi Bike trip data
# ----------------------------------

import zipfile
import shutil

def fetch_raw_citibike_data(year: int) -> Path:
    """
    Downloads and extracts Citi Bike data for a full year.

    - Downloads the yearly zip (e.g., 2023-citibike-tripdata.zip)
    - Extracts all monthly zip files inside it
    - Then extracts all CSVs inside each monthly zip into appropriate folders

    Returns:
        Path: Path to the unzipped yearly data directory
    """
    year_zip_url = f"https://s3.amazonaws.com/tripdata/{year}-citibike-tripdata.zip"
    year_zip_path = RAW_DATA_DIR / f"{year}-citibike-tripdata.zip"
    year_extract_dir = RAW_DATA_DIR / f"{year}-citibike-tripdata"

    # Download the full-year zip if not already present
    if not year_zip_path.exists():
        print(f"⬇️ Downloading {year}-citibike-tripdata.zip...")
        response = requests.get(year_zip_url)
        if response.status_code != 200:
            raise Exception(f"{year_zip_url} is not available")

        with open(year_zip_path, "wb") as f:
            f.write(response.content)
        print("✅ Downloaded full-year zip file.")

    # Unzip the full-year zip
    if not year_extract_dir.exists():
        print("📂 Extracting full-year zip...")
        with zipfile.ZipFile(year_zip_path, "r") as zip_ref:
            zip_ref.extractall(year_extract_dir)
        print("✅ Extracted full-year zip.")

    # Now unzip each monthly zip inside the extracted folder
    for monthly_zip in year_extract_dir.glob("*.zip"):
        month_folder_name = monthly_zip.stem  # e.g., 202301-citibike-tripdata
        month_folder_path = year_extract_dir / month_folder_name

        if not month_folder_path.exists():
            print(f"📦 Extracting {monthly_zip.name}...")
            month_folder_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(monthly_zip, "r") as zip_ref:
                zip_ref.extractall(month_folder_path)
            print(f"✅ Extracted to {month_folder_path.name}")

    return year_extract_dir





# ----------------------------------------
# 2. Filter and transform Citi Bike data
# ----------------------------------------

def filter_citibike_data(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Filter Citi Bike dataframe to retain rides starting from top 3 stations for the given year/month.

    Args:
        df (pd.DataFrame): Raw dataframe from CSV
        year (int): Year being processed
        month (int): Month being processed

    Returns:
        pd.DataFrame: Filtered dataframe with valid rides from top start stations
    """
    required_cols = [
        'ride_id', 'rideable_type', 'started_at', 'ended_at',
        'start_station_id', 'end_station_id',
        'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_casual'
    ]
    
    # Check all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {', '.join(missing_cols)}")

    # Parse timestamps
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')

    # Drop invalid rows
    df = df.dropna(subset=[
        'started_at', 'ended_at',
        'start_lat', 'start_lng', 'end_lat', 'end_lng', 'start_station_id'
    ])

    # Convert start_station_id to string or float if needed
    df['start_station_id'] = df['start_station_id'].astype(str)

    # Define top 3 start station IDs (as strings for consistency)
    top_station_ids = {'6140.05', '6948.10', '5329.03'}
    df = df[df['start_station_id'].isin(top_station_ids)]

    # Add year and month for traceability
    df['year'] = year
    df['month'] = month

    return df




# ------------------------------------------------------
# 3. Load and process multiple months of Citi Bike data
# ------------------------------------------------------

import zipfile
import requests
import shutil
from pathlib import Path
from typing import Optional, List
import pandas as pd
from src.config import RAW_DATA_DIR
from src.data_utils import filter_citibike_data

def load_and_process_citibike_data(year: int, months: Optional[List[int]] = None) -> pd.DataFrame:
    if months is None:
        months = list(range(1, 13))

    year_zip_name = f"{year}-citibike-tripdata.zip"
    year_zip_path = RAW_DATA_DIR / year_zip_name
    year_extract_path = RAW_DATA_DIR / f"{year}-citibike-tripdata"

    # 1. Download year zip if not present
    if not year_zip_path.exists():
        url = f"https://s3.amazonaws.com/tripdata/{year_zip_name}"
        print(f"⬇️ Downloading {year_zip_name} from {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(year_zip_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)
            print(f"✅ Downloaded {year_zip_name}")
        else:
            raise Exception(f"❌ Failed to download: {url} — Status {response.status_code}")

    # 2. Unzip year file
    if not year_extract_path.exists():
        print(f"📦 Extracting {year_zip_path.name}")
        with zipfile.ZipFile(year_zip_path, "r") as zip_ref:
            zip_ref.extractall(year_extract_path)

    # 3. Detect nested folder (actual monthly ZIPs may be here)
    nested_folders = list(year_extract_path.glob("*/"))
    if nested_folders:
        monthly_zip_dir = nested_folders[0]  # First folder inside
    else:
        monthly_zip_dir = year_extract_path  # Fallback to main

    print(f"\n📁 Looking for monthly zips in: {monthly_zip_dir}")

    all_months_df = []

    for month in months:
        month_zip_name = f"{year}{month:02}-citibike-tripdata.zip"
        month_zip_path = monthly_zip_dir / month_zip_name
        month_extract_folder = monthly_zip_dir / f"{year}{month:02}-citibike-tripdata"

        if not month_zip_path.exists():
            print(f"⚠️ Monthly zip not found: {month_zip_path}, skipping.")
            continue

        # 4. Extract monthly zip
        if not month_extract_folder.exists():
            print(f"📦 Extracting {month_zip_name}")
            with zipfile.ZipFile(month_zip_path, "r") as zip_ref:
                zip_ref.extractall(month_extract_folder)

        # 5. Read CSVs from extracted folder
        print(f"\n📁 Checking for CSVs in: {month_extract_folder}")
        csv_files = list(month_extract_folder.glob("*.csv"))
        if not csv_files:
            print(f"⚠️ No CSV files found in {month_extract_folder}, skipping.")
            continue

        monthly_dfs = []
        for csv_file in csv_files:
            try:
                print(f"🗂️ Reading {csv_file.name}")
                df = pd.read_csv(csv_file)
                df_filtered = filter_citibike_data(df, year, month)
                monthly_dfs.append(df_filtered)
            except Exception as e:
                print(f"❌ Failed to process {csv_file.name}: {e}")

        if monthly_dfs:
            month_df = pd.concat(monthly_dfs, ignore_index=True)
            all_months_df.append(month_df)
            print(f"✅ Finished processing for {year}-{month:02}")
        else:
            print(f"⚠️ No valid data for {year}-{month:02}")

    if not all_months_df:
        raise Exception(f"❌ No valid Citi Bike data found for year {year} and months {months}")

    combined_df = pd.concat(all_months_df, ignore_index=True)
    print(f"\n✅ All data loaded. Total records: {len(combined_df):,}")
    return combined_df










# ------------------------------------------
# 4. Fill missing hourly location ride slots
# ------------------------------------------

def fill_missing_rides_full_range(df, hour_col, location_col, rides_col):
    df[hour_col] = pd.to_datetime(df[hour_col])
    full_hours = pd.date_range(df[hour_col].min(), df[hour_col].max(), freq="h")
    all_locations = df[location_col].unique()

    complete = pd.DataFrame([
        (h, l) for h in full_hours for l in all_locations
    ], columns=[hour_col, location_col])

    df_merged = pd.merge(complete, df, on=[hour_col, location_col], how="left")
    df_merged[rides_col] = df_merged[rides_col].fillna(0).astype(int)
    return df_merged

# ----------------------------------------------------
# 5. Transform raw data to time-series hourly format
# ----------------------------------------------------

def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    rides["pickup_hour"] = rides["pickup_datetime"].dt.floor("h")
    agg = rides.groupby(["pickup_hour", "pickup_location_id"]).size().reset_index(name="rides")
    filled = fill_missing_rides_full_range(agg, "pickup_hour", "pickup_location_id", "rides")
    return filled.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)

# -----------------------------------------------------------------------
# 6. Create sliding window features and targets from time-series data
# -----------------------------------------------------------------------

def transform_ts_data_info_features_and_target_loop(
    df, feature_col="rides", window_size=12, step_size=1
):
    location_ids = df["pickup_location_id"].unique()
    transformed_data = []

    for location_id in location_ids:
        loc_df = df[df["pickup_location_id"] == location_id].reset_index(drop=True)
        values = loc_df[feature_col].values
        times = loc_df["pickup_hour"].values

        if len(values) <= window_size:
            print(f"Skipping {location_id} - Not enough data")
            continue

        rows = []
        for i in range(0, len(values) - window_size, step_size):
            features = values[i:i+window_size]
            target = values[i+window_size]
            timestamp = times[i+window_size]
            rows.append(np.append(features, [target, location_id, timestamp]))

        columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)] + [
            "target", "pickup_location_id", "pickup_hour"
        ]
        transformed = pd.DataFrame(rows, columns=columns)
        transformed_data.append(transformed)

    final_df = pd.concat(transformed_data, ignore_index=True)
    features = final_df[columns[:-2] + ["pickup_hour", "pickup_location_id"]]
    targets = final_df["target"]
    return features, targets

# ---------------------------------------------------------
# 7. Single-location version of the transformation above
# ---------------------------------------------------------

def transform_ts_data_info_features_and_target(
    df, feature_col="rides", window_size=12, step_size=1
):
    values = df[feature_col].values
    times = df["pickup_hour"].values
    location = df["pickup_location_id"].iloc[0]

    rows = []
    for i in range(0, len(values) - window_size, step_size):
        features = values[i:i+window_size]
        target = values[i+window_size]
        timestamp = times[i+window_size]
        rows.append(np.append(features, [target, location, timestamp]))

    columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)] + [
        "target", "pickup_location_id", "pickup_hour"
    ]
    df_transformed = pd.DataFrame(rows, columns=columns)
    features = df_transformed[columns[:-2] + ["pickup_hour", "pickup_location_id"]]
    targets = df_transformed["target"]
    return features, targets

# ----------------------------------------------------
# 8. Train-test split based on pickup_hour timestamp
# ----------------------------------------------------

def split_time_series_data(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = df[df["pickup_hour"] < cutoff_date]
    test = df[df["pickup_hour"] >= cutoff_date]
    return train.drop(columns=[target_column]), train[target_column], test.drop(columns=[target_column]), test[target_column]

# ----------------------------------------------------
# 9. Simulate batch fetch for 52 weeks ago (optional)
# ----------------------------------------------------

def fetch_batch_raw_data(from_date: Union[datetime, str], to_date: Union[datetime, str]) -> pd.DataFrame:
    if isinstance(from_date, str):
        from_date = datetime.fromisoformat(from_date)
    if isinstance(to_date, str):
        to_date = datetime.fromisoformat(to_date)

    from_hist = from_date - timedelta(weeks=52)
    to_hist = to_date - timedelta(weeks=52)

    months = list(set([from_hist.month, to_hist.month]))
    data = load_and_process_citibike_data(from_hist.year, months=months)
    data = data[(data["pickup_datetime"] >= from_hist) & (data["pickup_datetime"] < to_hist)]
    data["pickup_datetime"] += timedelta(weeks=52)

    return data.sort_values(["pickup_location_id", "pickup_datetime"]).reset_index(drop=True)

def transform_ts_data_info_features(
    df, feature_col="rides", window_size=12, step_size=1
):
    """
    Transforms time series data for all unique location IDs into a tabular format.
    The first `window_size` rows are used as features.
    The process slides down by `step_size` rows at a time to create the next set of features.
    Feature columns are named based on their hour offsets.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing time series data with 'pickup_hour' column.
        feature_col (str): The column name containing the values to use as features (default is "rides").
        window_size (int): The number of rows to use as features (default is 12).
        step_size (int): The number of rows to slide the window by (default is 1).

    Returns:
        pd.DataFrame: Features DataFrame with pickup_hour and location_id.
    """
    # Get all unique location IDs
    location_ids = df["pickup_location_id"].unique()
    # List to store transformed data for each location
    transformed_data = []

    # Loop through each location ID and transform the data
    for location_id in location_ids:
        try:
            # Filter the data for the given location ID
            location_data = df[df["pickup_location_id"] == location_id].reset_index(
                drop=True
            )

            # Extract the feature column and pickup_hour as NumPy arrays
            values = location_data[feature_col].values
            times = location_data["pickup_hour"].values

            # Ensure there are enough rows to create at least one window
            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            # Create the tabular data using a sliding window approach
            rows = []
            for i in range(0, len(values) - window_size, step_size):
                # The first `window_size` values are features
                features = values[i : i + window_size]
                # Get the corresponding target timestamp
                target_time = times[i + window_size]
                row = np.append(features, [location_id, target_time])
                rows.append(row)

            # Convert the list of rows into a DataFrame
            feature_columns = [
                f"{feature_col}_t-{window_size - i}" for i in range(window_size)
            ]
            all_columns = feature_columns + ["pickup_location_id", "pickup_hour"]
            transformed_df = pd.DataFrame(rows, columns=all_columns)

            # Append the transformed data to the list
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping location_id {location_id}: {str(e)}")

    # Combine all transformed data into a single DataFrame
    if not transformed_data:
        raise ValueError(
            "No data could be transformed. Check if input DataFrame is empty or window size is too large."
        )

    final_df = pd.concat(transformed_data, ignore_index=True)

    return final_df

from pathlib import Path
import pandas as pd
import zipfile


def load_and_process_citibike_data_from_local(year: int, months: list, base_path: str = "data/citibike") -> pd.DataFrame:
    """
    Load Citi Bike data directly from locally extracted or zipped CSV files.

    Supports files like:
    - 202504-citibike-tripdata.csv
    - 202504-citibike-tripdata.csv.zip
    - 202504-citibike-tripdata.zip

    Parameters:
        year (int): The year of the data.
        months (list of int): The months to load (e.g., [1, 2, 3]).
        base_path (str): Base folder where CSVs or ZIPs are stored.

    Returns:
        pd.DataFrame: Concatenated and cleaned dataframe for the given months.
    """
    all_months_data = []
    data_dir = Path(base_path)

    for month in months:
        month_str = f"{year}{month:02d}"
        csv_files = (
            list(data_dir.glob(f"{month_str}-citibike-tripdata*.csv")) +
            list(data_dir.glob(f"{month_str}-citibike-tripdata*.csv.zip")) +
            list(data_dir.glob(f"{month_str}-citibike-tripdata.zip"))
        )

        if not csv_files:
            raise FileNotFoundError(f"❌ No CSV/ZIP files found for {month_str} in {data_dir}")

        for file_path in csv_files:
            if file_path.suffix == ".zip":
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for inner_file in zip_ref.namelist():
                        if inner_file.endswith(".csv"):
                            with zip_ref.open(inner_file) as f:
                                df = pd.read_csv(f)
                                df['source_file'] = file_path.name
                                all_months_data.append(df)
            elif file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
                df['source_file'] = file_path.name
                all_months_data.append(df)
            elif file_path.suffixes[-2:] == [".csv", ".zip"]:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    csv_name = zip_ref.namelist()[0]
                    with zip_ref.open(csv_name) as f:
                        df = pd.read_csv(f)
                        df['source_file'] = file_path.name
                        all_months_data.append(df)

    df_all = pd.concat(all_months_data, ignore_index=True)
    df_all['started_at'] = pd.to_datetime(df_all['started_at'], errors='coerce')
    df_all['start_station_id'] = df_all['start_station_id'].astype(str)

    return df_all
