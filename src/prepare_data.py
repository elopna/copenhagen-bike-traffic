"""Data preparation: loading, cleaning, and train/test splitting."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from src.schema_detect import infer_schema, validate_schema
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


def load_and_detect_schema(
    rides_path: Path, total_rides_path: Path, road_info_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """
    Load all three CSVs and detect their schemas.
    
    Returns
    -------
    rides_df, total_rides_df, road_info_df, rides_schema, total_schema, road_schema
    """
    logger.info("Loading CSV files...")
    
    rides_df = pd.read_csv(rides_path)
    logger.info(f"Loaded rides.csv: {rides_df.shape[0]:,} rows, {rides_df.shape[1]} columns")
    
    total_rides_df = pd.read_csv(total_rides_path)
    logger.info(f"Loaded total_rides.csv: {total_rides_df.shape[0]:,} rows, {total_rides_df.shape[1]} columns")
    
    road_info_df = pd.read_csv(road_info_path)
    logger.info(f"Loaded road_info.csv: {road_info_df.shape[0]:,} rows, {road_info_df.shape[1]} columns")
    
    # Detect schemas
    logger.info("\n" + "=" * 60)
    rides_schema = infer_schema(rides_df, "rides.csv")
    
    logger.info("\n" + "=" * 60)
    total_schema = infer_schema(total_rides_df, "total_rides.csv")
    
    logger.info("\n" + "=" * 60)
    road_schema = infer_schema(road_info_df, "road_info.csv")
    logger.info("=" * 60 + "\n")
    
    # Validate schemas
    validate_schema(rides_schema, ["time", "site_id", "count"], "rides.csv")
    validate_schema(total_schema, ["time", "site_id", "count"], "total_rides.csv")
    validate_schema(road_schema, ["site_id", "lat", "lon"], "road_info.csv")
    
    return rides_df, total_rides_df, road_info_df, rides_schema, total_schema, road_schema


def prepare_temporal_features(df: pd.DataFrame, time_col: str, time_interval_col: str = None) -> pd.DataFrame:
    """
    Add temporal features: date, hour, dow, month, is_weekend.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe with time column
    time_col : str
        Name of date column (e.g., 'date')
    time_interval_col : str, optional
        Name of time interval column (e.g., 'time' with format "00-01", "19-20")
        
    Returns
    -------
    DataFrame
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Parse date column
    df["date_parsed"] = pd.to_datetime(df[time_col], errors="coerce")
    
    # Remove rows with invalid dates
    invalid_count = df["date_parsed"].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count} rows with invalid dates")
        df = df[df["date_parsed"].notna()].copy()
    
    # Parse hour from time interval column if provided (e.g., "00-01" -> 0, "19-20" -> 19)
    if time_interval_col and time_interval_col in df.columns:
        logger.info(f"Parsing hour from time interval column: {time_interval_col}")
        
        # Validate format "HH-HH"
        sample = df[time_interval_col].iloc[0]
        if not isinstance(sample, str) or "-" not in str(sample):
            raise ValueError(
                f"Time interval column '{time_interval_col}' has unexpected format. "
                f"Expected 'HH-HH' (e.g., '00-01', '19-20'), got: {sample}"
            )
        
        # Extract start hour from interval "HH-HH" format
        df["hour"] = df[time_interval_col].astype(str).str.split("-").str[0].astype(int)
        
        # Validate hour range
        if (df["hour"] < 0).any() or (df["hour"] > 23).any():
            raise ValueError(
                f"Parsed hour values out of range [0-23]. "
                f"Found: min={df['hour'].min()}, max={df['hour'].max()}"
            )
        
        # Create full timestamp = date + hour
        df["timestamp"] = pd.to_datetime(df["date_parsed"]) + pd.to_timedelta(df["hour"], unit="h")
    else:
        # No time interval, use date as timestamp and hour will be 0
        df["timestamp"] = df["date_parsed"]
        df["hour"] = df["timestamp"].dt.hour
    
    # Extract temporal features
    df["date"] = df["timestamp"].dt.date
    df["dow"] = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    
    # Drop temporary column
    df = df.drop(columns=["date_parsed"])
    
    return df


def extract_temporal_features_from_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from existing timestamp column.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with timestamp column
        
    Returns
    -------
    DataFrame
        DataFrame with added temporal features: date, hour, dow, month, is_weekend
    """
    df = df.copy()
    
    # Extract temporal features from timestamp
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    
    return df


def aggregate_to_hourly(df: pd.DataFrame, site_col: str, count_col: str) -> pd.DataFrame:
    """
    Aggregate to hourly frequency if not already.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe with timestamp, site_id, and count
    site_col : str
        Name of site ID column
    count_col : str
        Name of count column
        
    Returns
    -------
    DataFrame
        Aggregated dataframe at hourly frequency (without temporal features - need to be re-extracted)
    """
    # Round timestamp to hour
    df["timestamp_hour"] = df["timestamp"].dt.floor("H")
    
    # Group by site and hour, sum counts
    agg_df = (
        df.groupby([site_col, "timestamp_hour"], as_index=False)
        .agg({count_col: "sum"})
        .rename(columns={"timestamp_hour": "timestamp"})
    )
    
    logger.info(f"Aggregated to hourly: {agg_df.shape[0]:,} rows")
    
    return agg_df


def clean_data(df: pd.DataFrame, count_col: str, site_col: str) -> pd.DataFrame:
    """
    Data quality control: remove negatives and outliers.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    count_col : str
        Name of count column
    site_col : str
        Name of site ID column
        
    Returns
    -------
    DataFrame
        Cleaned dataframe
    """
    df = df.copy()
    original_count = len(df)
    
    # Remove negative counts
    negatives = (df[count_col] < 0).sum()
    if negatives > 0:
        logger.warning(f"Removing {negatives} rows with negative counts")
        df = df[df[count_col] >= 0].copy()
    
    # Remove outliers: > 99.9th percentile per site
    outliers_removed = 0
    sites = df[site_col].unique()
    
    for site in sites:
        site_mask = df[site_col] == site
        site_data = df.loc[site_mask, count_col]
        
        if len(site_data) > 0:
            threshold = site_data.quantile(0.999)
            outlier_mask = site_mask & (df[count_col] > threshold)
            outliers_count = outlier_mask.sum()
            
            if outliers_count > 0:
                outliers_removed += outliers_count
                df = df[~outlier_mask].copy()
    
    if outliers_removed > 0:
        logger.warning(f"Removed {outliers_removed} outliers (>99.9th percentile per site)")
    
    logger.info(f"Data cleaning: {original_count:,} -> {len(df):,} rows ({original_count - len(df):,} removed)")
    
    return df


def join_with_road_info(
    df: pd.DataFrame,
    road_info: pd.DataFrame,
    site_col: str,
    road_site_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    """
    Join rides with road_info to get coordinates.
    
    Parameters
    ----------
    df : DataFrame
        Rides dataframe
    road_info : DataFrame
        Road info dataframe with coordinates
    site_col : str
        Name of site ID column in rides (e.g., road_name)
    road_site_col : str
        Name of site ID column in road_info (e.g., road_name)
    lat_col : str
        Name of latitude column in road_info
    lon_col : str
        Name of longitude column in road_info
        
    Returns
    -------
    DataFrame
        Joined dataframe with lat/lon
    """
    # Prepare road_info
    road_info_clean = road_info[[road_site_col, lat_col, lon_col]].copy()
    
    # Remove invalid coordinates first
    road_info_clean = road_info_clean[
        road_info_clean[lat_col].notna() & road_info_clean[lon_col].notna()
    ].copy()
    
    # Remove duplicates (keep first)
    road_info_clean = road_info_clean.drop_duplicates(subset=[road_site_col], keep="first")
    
    logger.info(f"Road info: {len(road_info_clean)} unique sites with valid coordinates")
    logger.info(f"Sample road_info site IDs: {road_info_clean[road_site_col].head().tolist()}")
    logger.info(f"Sample rides site IDs: {df[site_col].unique()[:5].tolist()}")
    
    # Rename for merge
    road_info_clean = road_info_clean.rename(
        columns={road_site_col: "join_key", lat_col: "lat", lon_col: "lon"}
    )
    
    # Join
    original_count = len(df)
    df_with_join_key = df.copy()
    df_with_join_key["join_key"] = df_with_join_key[site_col]
    
    df_joined = df_with_join_key.merge(road_info_clean, on="join_key", how="inner")
    df_joined = df_joined.drop(columns=["join_key"])
    
    logger.info(f"After join with road_info: {len(df_joined):,} rows ({original_count - len(df_joined):,} dropped)")
    
    if len(df_joined) == 0:
        # Get more debug info
        rides_sites = set(df[site_col].unique())
        road_sites = set(road_info[road_site_col].dropna().unique())
        common = rides_sites & road_sites
        
        raise ValueError(
            f"No rows remaining after join. Check that site IDs match between rides and road_info.\n"
            f"Rides has {len(rides_sites)} unique site IDs\n"
            f"Road info has {len(road_sites)} unique site IDs\n"
            f"Common site IDs: {len(common)}\n"
            f"Sample site IDs from rides: {list(rides_sites)[:5]}\n"
            f"Sample site IDs from road_info: {list(road_sites)[:5]}\n"
            f"Join columns: rides[{site_col}] <-> road_info[{road_site_col}]"
        )
    
    return df_joined


def convert_utm_to_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert UTM coordinates to lat/lon if needed.
    
    If coordinates look like UTM (large values around 700k for x, 6.1M for y),
    convert from UTM zone 32N to WGS84.
    
    Assumes:
    - 'lon' column contains x_coord (easting) from UTM
    - 'lat' column contains y_coord (northing) from UTM
    """
    from pyproj import Transformer
    
    df = df.copy()
    
    # Check if coordinates look like UTM (heuristic: values > 100,000)
    sample_lat = df["lat"].iloc[0]
    sample_lon = df["lon"].iloc[0]
    
    # UTM coordinates are typically 6-7 digits
    if sample_lat > 100_000 or sample_lon > 100_000:
        logger.info("Coordinates appear to be UTM, converting to WGS84 lat/lon...")
        logger.info(f"Sample UTM: x={sample_lon:.0f}, y={sample_lat:.0f}")
        
        # Copenhagen is in UTM zone 32N (EPSG:25832)
        # Convert to WGS84 (EPSG:4326)
        # always_xy=True means input is (x, y) = (easting, northing)
        transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        
        # Transform: (x_coord=easting, y_coord=northing) -> (lon, lat)
        # df["lon"] contains x_coord (easting)
        # df["lat"] contains y_coord (northing)
        lon_wgs84, lat_wgs84 = transformer.transform(df["lon"].values, df["lat"].values)
        
        df["lat"] = lat_wgs84
        df["lon"] = lon_wgs84
        
        logger.info(f"Converted to WGS84. Sample: lat={df['lat'].iloc[0]:.6f}, lon={df['lon'].iloc[0]:.6f}")
    else:
        logger.info("Coordinates appear to already be in lat/lon format (WGS84)")
    
    return df


def temporal_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test using temporal holdout.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe with timestamp column
    test_size : float
        Fraction of data to use for test (default 0.2 = 20%)
        
    Returns
    -------
    train_df, test_df
    """
    df = df.sort_values("timestamp").copy()
    
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    coverage_days = (max_date - min_date).days
    
    logger.info(f"Date range: {min_date} to {max_date} ({coverage_days} days)")
    
    # Use last test_size fraction of unique timestamps
    unique_timestamps = df["timestamp"].unique()
    unique_timestamps = np.sort(unique_timestamps)
    n_test = int(len(unique_timestamps) * test_size)
    test_start = unique_timestamps[-n_test]
    
    test_mask = df["timestamp"] >= test_start
    
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    
    train_df["split"] = "train"
    test_df["split"] = "test"
    
    test_days = (test_df["timestamp"].max() - test_df["timestamp"].min()).days
    
    logger.info(f"Test set: last {test_size*100:.0f}% of timestamps ({n_test} unique times, ~{test_days} days)")
    logger.info(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    logger.info(f"Train: {len(train_df):,} rows ({len(train_df) / len(df) * 100:.1f}%), {train_df['site_id'].nunique()} unique sites")
    logger.info(f"Test:  {len(test_df):,} rows ({len(test_df) / len(df) * 100:.1f}%), {test_df['site_id'].nunique()} unique sites")
    
    # Check if any sites are missing in train or test
    train_sites = set(train_df['site_id'].unique())
    test_sites = set(test_df['site_id'].unique())
    only_in_train = train_sites - test_sites
    only_in_test = test_sites - train_sites
    
    if only_in_train:
        logger.warning(f"Sites present only in train (not in test): {sorted(only_in_train)}")
    if only_in_test:
        logger.warning(f"Sites present only in test (not in train): {sorted(only_in_test)}")
    
    return train_df, test_df


def prepare_data_pipeline(
    data_dir: Path,
    artifacts_dir: Path,
    freq: str = "hour",
    test_size: float = 0.2,
) -> None:
    """
    Full data preparation pipeline.
    
    Parameters
    ----------
    data_dir : Path
        Directory with rides.csv, total_rides.csv, road_info.csv
    artifacts_dir : Path
        Directory to save outputs
    freq : str
        Frequency (currently only "hour" supported)
    test_size : float
        Fraction of data for test set (default 0.2 = 20%)
    """
    ensure_dir(artifacts_dir)
    
    # Load and detect schemas
    rides_df, total_rides_df, road_info_df, rides_schema, total_schema, road_schema = (
        load_and_detect_schema(
            data_dir / "rides.csv",
            data_dir / "total_rides.csv",
            data_dir / "road_info.csv",
        )
    )
    
    # Use total_rides.csv (already filtered to full counts)
    logger.info("\nUsing total_rides.csv for modeling (full counts only)")
    df = total_rides_df.copy()
    schema = total_schema
    
    # Prepare temporal features (combine date + time interval)
    logger.info("\nPreparing temporal features...")
    # Check if there's a separate 'time' column for hour intervals
    # The 'time' column should contain hour intervals like "00-01", "19-20"
    # while schema["time"] points to the date column
    time_interval_col = None
    if "time" in df.columns:
        # Check if 'time' is different from the date column
        if schema["time"] != "time":
            # 'time' is a separate column from the date column
            time_interval_col = "time"
            logger.info(f"Found time interval column: {time_interval_col}")
        else:
            logger.info("Column 'time' is the date column, no separate interval column")
    
    df = prepare_temporal_features(df, schema["time"], time_interval_col=time_interval_col)
    
    # Aggregate to hourly if needed
    if freq == "hour":
        # Aggregate (this removes temporal features, keeping only timestamp, site_id, count)
        df = aggregate_to_hourly(df, schema["site_id"], schema["count"])
        # Re-extract temporal features from the aggregated timestamp
        df = extract_temporal_features_from_timestamp(df)
    else:
        raise ValueError(f"Unsupported frequency: {freq}. Only 'hour' is supported.")
    
    # Rename columns to standard names
    df = df.rename(columns={schema["site_id"]: "site_id", schema["count"]: "count"})
    
    # Clean data
    logger.info("\nCleaning data...")
    df = clean_data(df, "count", "site_id")
    
    # Join with road_info
    # Note: rides.csv has road_name, road_info.csv has both road_name and road_id
    # We join by road_name (site_id in our standardized schema)
    logger.info("\nJoining with road_info...")
    
    # Check if we need to use the original column names for joining
    # (before renaming to "site_id")
    df_join_col = schema["site_id"]  # Original column name from rides
    road_join_col = road_schema["site_id"]  # Original column name from road_info
    
    # If we already renamed, use "site_id"
    if "site_id" in df.columns and df_join_col not in df.columns:
        df_join_col = "site_id"
    
    df = join_with_road_info(
        df,
        road_info_df,
        df_join_col,
        road_join_col,
        road_schema["lat"],
        road_schema["lon"],
    )
    
    # Convert UTM to lat/lon if needed
    df = convert_utm_to_latlon(df)
    
    # Train/test split
    logger.info("\nSplitting train/test...")
    train_df, test_df = temporal_train_test_split(df, test_size)
    
    # Save artifacts
    logger.info("\nSaving artifacts...")
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    all_df.to_parquet(artifacts_dir / "clean.parquet", index=False)
    logger.info(f"Saved: {artifacts_dir / 'clean.parquet'}")
    
    train_df.to_parquet(artifacts_dir / "train.parquet", index=False)
    logger.info(f"Saved: {artifacts_dir / 'train.parquet'}")
    
    test_df.to_parquet(artifacts_dir / "test.parquet", index=False)
    logger.info(f"Saved: {artifacts_dir / 'test.parquet'}")
    
    logger.info("\nData preparation complete!")
    logger.info(f"Total sites: {df['site_id'].nunique()}")
    logger.info(f"Total hours: {df['timestamp'].dt.floor('H').nunique()}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

