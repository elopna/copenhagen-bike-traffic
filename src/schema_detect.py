"""Automatic schema detection for Copenhagen bike counter CSV files."""

import logging
from typing import Dict, Optional, List

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def _find_column(
    df: pd.DataFrame, keywords: List[str], numeric_only: bool = False, exact: bool = False
) -> Optional[str]:
    """
    Find column matching any of the keywords (case-insensitive).
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    keywords : list of str
        List of keywords to search for in column names
    numeric_only : bool
        If True, only consider numeric columns
    exact : bool
        If True, require exact match (ignoring case and underscores)
        
    Returns
    -------
    str or None
        Matched column name or None
    """
    candidates = df.columns.tolist()
    
    if numeric_only:
        candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    
    # First pass: try exact matches (if requested)
    if exact:
        for col in candidates:
            col_normalized = col.lower().replace("_", "").replace(" ", "")
            for keyword in keywords:
                keyword_normalized = keyword.lower().replace("_", "").replace(" ", "")
                if col_normalized == keyword_normalized:
                    return col
    
    # Second pass: substring matches
    for col in candidates:
        col_lower = col.lower()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                return col
    return None


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find time column by trying to parse as datetime.
    
    Returns column with highest fraction of successfully parsed dates.
    """
    time_keywords = ["timestamp", "datetime", "time", "date"]
    candidates = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in time_keywords):
            # Try to parse
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid_frac = parsed.notna().mean()
            if valid_frac > 0.5:  # At least 50% valid dates
                candidates.append((col, valid_frac))
    
    if candidates:
        # Return column with highest valid fraction
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    return None


def infer_schema(df: pd.DataFrame, file_description: str = "") -> Dict[str, Optional[str]]:
    """
    Automatically infer schema for bike counter CSV.
    
    Detects:
    - time: timestamp/datetime/date column
    - site_id: site/counter/station/location/id column
    - count: numeric column with count/rides/bikes/traffic/volume
    - lat: latitude column (numeric)
    - lon: longitude column (numeric)
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    file_description : str
        Description for logging (e.g., "rides.csv")
        
    Returns
    -------
    dict
        Schema dictionary with keys: time, site_id, count, lat, lon
        Values are column names or None if not found
    """
    schema = {}
    
    # Find time column
    schema["time"] = _find_time_column(df)
    
    # Find site ID column
    # Try exact match for road_name first (important for joining between tables)
    schema["site_id"] = _find_column(df, ["road_name"], exact=True)
    if not schema["site_id"]:
        # Fallback to broader search
        schema["site_id"] = _find_column(
            df, ["road_name", "site", "counter", "station", "location", "road_id", "id"]
        )
    
    # Find count column (numeric only)
    schema["count"] = _find_column(
        df, ["count", "rides", "bikes", "traffic", "volume", "n"], numeric_only=True
    )
    
    # Find latitude column (numeric only)
    # Note: y_coord in UTM is northing (latitude-like)
    schema["lat"] = _find_column(
        df, ["lat", "latitude", "y_coord", "y"], numeric_only=True
    )
    
    # Find longitude column (numeric only)
    # Note: x_coord in UTM is easting (longitude-like)
    schema["lon"] = _find_column(
        df, ["lon", "lng", "long", "longitude", "x_coord", "x"], numeric_only=True
    )
    
    # Log the detected schema
    if file_description:
        logger.info(f"Schema detected for {file_description}:")
    else:
        logger.info("Schema detected:")
    
    for key, col in schema.items():
        if col:
            logger.info(f"  {key:10s} -> {col}")
        else:
            logger.info(f"  {key:10s} -> NOT FOUND")
    
    return schema


def validate_schema(
    schema: Dict[str, Optional[str]],
    required_fields: List[str],
    file_description: str = "",
) -> None:
    """
    Validate that required fields are present in schema.
    
    Parameters
    ----------
    schema : dict
        Schema dictionary from infer_schema
    required_fields : list of str
        List of required field names
    file_description : str
        Description for error message
        
    Raises
    ------
    ValueError
        If any required field is missing
    """
    missing = [field for field in required_fields if not schema.get(field)]
    
    if missing:
        error_msg = f"Missing required columns in {file_description}: {missing}\n"
        error_msg += "Please ensure the CSV file contains columns matching these patterns:\n"
        
        patterns = {
            "time": "timestamp, datetime, time, or date",
            "site_id": "site, counter, station, location, road_name, road_id, or id",
            "count": "count, rides, bikes, traffic, volume, or n (numeric)",
            "lat": "lat, latitude, y_coord, or y (numeric)",
            "lon": "lon, lng, longitude, x_coord, or x (numeric)",
        }
        
        for field in missing:
            if field in patterns:
                error_msg += f"  {field}: {patterns[field]}\n"
        
        raise ValueError(error_msg)

