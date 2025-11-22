"""Advanced features: weather, holidays, and lagged features."""

import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_weather_features(
    df: pd.DataFrame, weather_path: Path, location_id: int = 0
) -> pd.DataFrame:
    """
    Add weather features from Open-Meteo data.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with timestamp column
    weather_path : Path
        Path to open-meteo CSV file
    location_id : int
        Location ID to use from weather file (default 0)
        
    Returns
    -------
    DataFrame
        DataFrame with added weather features
    """
    df = df.copy()
    
    logger.info(f"Loading weather data from {weather_path}")
    
    # Read weather file (skip first metadata section)
    weather_df = pd.read_csv(weather_path, skiprows=14)
    
    # Filter by location_id
    weather_df = weather_df[weather_df["location_id"] == location_id].copy()
    
    # Parse time
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    weather_df = weather_df.rename(columns={"time": "timestamp"})
    
    # Clean column names
    weather_df = weather_df.rename(columns={
        "temperature_2m (°C)": "temperature",
        "wind_speed_10m (km/h)": "wind_speed",
        "rain (mm)": "rain",
        "snowfall (cm)": "snowfall",
    })
    
    # Select relevant columns
    weather_cols = ["timestamp", "temperature", "wind_speed", "rain", "snowfall"]
    weather_df = weather_df[weather_cols]
    
    logger.info(f"Loaded {len(weather_df):,} weather records")
    
    # Merge with main dataframe
    df = df.merge(weather_df, on="timestamp", how="left")
    
    # Fill missing values with median (for edge cases)
    for col in ["temperature", "wind_speed", "rain", "snowfall"]:
        if df[col].isna().any():
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            logger.warning(f"Filled {n_missing} missing {col} values with median: {median_val:.2f}")
    
    logger.info("Added weather features: temperature, wind_speed, rain, snowfall")
    
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Danish holiday features.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with timestamp column
        
    Returns
    -------
    DataFrame
        DataFrame with is_holiday column
    """
    df = df.copy()
    
    # Extract date
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    
    # Danish holidays
    dk_holidays = holidays.country_holidays("DK")
    
    # Check if date is holiday
    df["is_holiday"] = df["date"].apply(lambda d: d in dk_holidays).astype(int)
    
    # Drop temporary column
    df = df.drop(columns=["date"])
    
    n_holidays = df["is_holiday"].sum()
    logger.info(f"Added holiday features: {n_holidays:,} holiday rows ({n_holidays/len(df)*100:.2f}%)")
    
    return df


def add_lagged_features(df: pd.DataFrame, target_col: str = "count") -> pd.DataFrame:
    """
    Add lagged and rolling features.
    
    NOTE: These features should ONLY be used for:
    - OOF predictions in CV (each fold computes lags from its train data)
    - Test predictions (compute lags from full train data)
    
    Do NOT use for regular train data (data leakage!).
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with site_id, timestamp, and target column
    target_col : str
        Name of target column to create lags from
        
    Returns
    -------
    DataFrame
        DataFrame with lagged features
    """
    df = df.copy()
    
    # Sort by site and time
    df = df.sort_values(["site_id", "timestamp"])
    
    # Lag 24 hours (yesterday same hour)
    df["count_lag_24h"] = df.groupby("site_id")[target_col].shift(24)
    
    # Rolling mean 7 days (168 hours)
    df["count_rolling_mean_7d"] = (
        df.groupby("site_id")[target_col]
        .transform(lambda x: x.rolling(window=168, min_periods=24).mean())
    )
    
    # Fill NaN with site median (for first days where lags don't exist)
    site_medians = df.groupby("site_id")[target_col].transform("median")
    df["count_lag_24h"] = df["count_lag_24h"].fillna(site_medians)
    df["count_rolling_mean_7d"] = df["count_rolling_mean_7d"].fillna(site_medians)
    
    n_lag_missing = df["count_lag_24h"].isna().sum()
    n_roll_missing = df["count_rolling_mean_7d"].isna().sum()
    
    logger.info(f"Added lagged features: count_lag_24h, count_rolling_mean_7d")
    if n_lag_missing > 0 or n_roll_missing > 0:
        logger.warning(
            f"Filled missing lags: {n_lag_missing} lag_24h, {n_roll_missing} rolling_7d"
        )
    
    return df


def advanced_features_pipeline(
    artifacts_dir: Path,
    weather_path: Path,
    location_id: int = 0,
    add_lags: bool = False,
) -> None:
    """
    Add advanced features to train and test datasets.
    
    Parameters
    ----------
    artifacts_dir : Path
        Directory with features.parquet and features_test.parquet
    weather_path : Path
        Path to open-meteo weather CSV
    location_id : int
        Location ID from weather file (default 0 = first location)
    add_lags : bool
        Whether to add lagged features (default False - handle separately in models)
    """
    logger.info("=" * 60)
    logger.info("Advanced Features Pipeline")
    logger.info("=" * 60)
    
    # Load data
    train_df = pd.read_parquet(artifacts_dir / "features.parquet")
    test_df = pd.read_parquet(artifacts_dir / "features_test.parquet")
    
    logger.info(f"Loaded train: {train_df.shape[0]:,} rows, {train_df.shape[1]} cols")
    logger.info(f"Loaded test:  {test_df.shape[0]:,} rows, {test_df.shape[1]} cols")
    
    # Add weather features
    logger.info("\nAdding weather features...")
    train_df = add_weather_features(train_df, weather_path, location_id)
    test_df = add_weather_features(test_df, weather_path, location_id)
    
    # Add holiday features
    logger.info("\nAdding holiday features...")
    train_df = add_holiday_features(train_df)
    test_df = add_holiday_features(test_df)
    
    # Add lagged features (optional - usually added in model training)
    if add_lags:
        logger.info("\nAdding lagged features...")
        train_df = add_lagged_features(train_df, "count")
        test_df = add_lagged_features(test_df, "count")
    else:
        logger.info("\nSkipping lagged features (will be added during model training)")
    
    # Save
    train_df.to_parquet(artifacts_dir / "features.parquet", index=False)
    test_df.to_parquet(artifacts_dir / "features_test.parquet", index=False)
    
    logger.info(f"\nSaved train: {train_df.shape[0]:,} rows, {train_df.shape[1]} cols")
    logger.info(f"Saved test:  {test_df.shape[0]:,} rows, {test_df.shape[1]} cols")
    logger.info("\nAdvanced features added successfully!")
