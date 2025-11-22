"""Feature engineering: temporal, spatial, and neighbor-based features."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import h3
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def add_h3_features(df: pd.DataFrame, resolution: int = 8) -> pd.DataFrame:
    """
    Add H3 hexagonal index at specified resolution.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with lat, lon columns
    resolution : int
        H3 resolution (8 = ~0.46 km² hexagons)
        
    Returns
    -------
    DataFrame
        DataFrame with h3_r{resolution} column
    """
    df = df.copy()
    
    col_name = f"h3_r{resolution}"
    # h3 v4.x uses latlng_to_cell instead of geo_to_h3
    df[col_name] = df.apply(
        lambda row: h3.latlng_to_cell(row["lat"], row["lon"], resolution), axis=1
    )
    
    logger.info(f"Added H3 resolution {resolution}: {df[col_name].nunique()} unique cells")
    
    return df


def add_geocenter_distance(df: pd.DataFrame, center_lat: float = None, center_lon: float = None) -> tuple:
    """
    Add distance to geocenter of all sites.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with lat, lon columns
    center_lat : float, optional
        Latitude of geocenter (if None, computed from df)
    center_lon : float, optional
        Longitude of geocenter (if None, computed from df)
        
    Returns
    -------
    tuple
        (DataFrame with dist_to_center column, center_lat, center_lon)
    """
    df = df.copy()
    
    # Compute geocenter only if not provided (i.e., for train)
    if center_lat is None or center_lon is None:
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
        logger.info(f"Computing geocenter: lat={center_lat:.6f}, lon={center_lon:.6f}")
    else:
        logger.info(f"Using provided geocenter: lat={center_lat:.6f}, lon={center_lon:.6f}")
    
    # Haversine distance
    def haversine(lat1, lon1, lat2, lon2):
        """Haversine distance in km."""
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    df["dist_to_center"] = haversine(df["lat"], df["lon"], center_lat, center_lon)
    
    return df, center_lat, center_lon


def compute_neighbor_graph(
    df: pd.DataFrame, k: int = 10
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Build KNN neighbor graph based on site coordinates.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with site_id, lat, lon
    k : int
        Number of nearest neighbors
        
    Returns
    -------
    neighbor_indices : dict
        {site_id: array of k neighbor site_ids}
    neighbor_distances : dict
        {site_id: array of k distances in km}
    """
    # Get unique sites
    sites = df[["site_id", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    
    logger.info(f"Building KNN graph for {len(sites)} sites with k={k}")
    
    # Fit KNN (use lat/lon directly for simplicity, good enough for small regions)
    coords = sites[["lat", "lon"]].values
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="haversine", algorithm="ball_tree")
    nbrs.fit(np.radians(coords))
    
    distances, indices = nbrs.kneighbors(np.radians(coords))
    
    # Convert to km
    distances = distances * 6371
    
    # Remove self (first neighbor is always self)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # Build dictionaries
    neighbor_indices = {}
    neighbor_distances = {}
    
    for i, site_id in enumerate(sites["site_id"]):
        neighbor_site_ids = sites.iloc[indices[i]]["site_id"].values
        neighbor_indices[site_id] = neighbor_site_ids
        neighbor_distances[site_id] = distances[i]
    
    logger.info(f"Neighbor graph built. Mean distance to nearest neighbor: {distances[:, 0].mean():.2f} km")
    
    return neighbor_indices, neighbor_distances


def add_neighbor_density_features(
    df: pd.DataFrame, neighbor_distances: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Add neighbor density features: count of neighbors within radius thresholds.
    
    NOTE: Removed neighbor_250m/500m/1km features as SHAP showed 0 importance.
    Keeping function for backward compatibility.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with site_id
    neighbor_distances : dict
        {site_id: array of distances to k neighbors}
        
    Returns
    -------
    DataFrame
        DataFrame (unchanged - features removed due to 0 importance)
    """
    df = df.copy()
    
    # Features removed - SHAP importance = 0
    logger.info("Neighbor density features removed (0 SHAP importance)")
    
    return df


def add_neighbor_aggregate_features(
    df: pd.DataFrame, neighbor_distances: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Add aggregate features from neighbor distances.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with site_id
    neighbor_distances : dict
        {site_id: array of distances to k neighbors}
        
    Returns
    -------
    DataFrame
        DataFrame with neighbor_mean_dist, neighbor_median_dist columns
    """
    df = df.copy()
    
    df["neighbor_mean_dist"] = df["site_id"].map(
        lambda site_id: neighbor_distances.get(site_id, np.array([np.nan])).mean()
    )
    
    df["neighbor_median_dist"] = df["site_id"].map(
        lambda site_id: np.median(neighbor_distances.get(site_id, np.array([np.nan])))
    )
    
    logger.info("Added neighbor aggregate distance features")
    
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encoding of hour and day of year as sin/cos.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with hour and timestamp columns
        
    Returns
    -------
    DataFrame
        DataFrame with hour_sin, hour_cos, year_day_sin, year_day_cos columns
    """
    df = df.copy()
    
    # Hour cyclical features (24-hour cycle)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # Day of year cyclical features (365-day cycle)
    # dayofyear: 1-365 (or 1-366 for leap years)
    df["year_day"] = df["timestamp"].dt.dayofyear
    df["year_day_sin"] = np.sin(2 * np.pi * df["year_day"] / 365.25)
    df["year_day_cos"] = np.cos(2 * np.pi * df["year_day"] / 365.25)
    
    # Drop temporary column
    df = df.drop(columns=["year_day"])
    
    logger.info("Added cyclical features: hour (sin/cos), year_day (sin/cos)")
    
    return df


def add_h3_onehot(df: pd.DataFrame, h3_col: str = "h3_r8") -> pd.DataFrame:
    """
    Add one-hot encoding for H3 column (for XGBoost).
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with h3 column
    h3_col : str
        Name of H3 column to encode
        
    Returns
    -------
    DataFrame
        DataFrame with h3_r8_enc_* columns
    """
    df = df.copy()
    
    # One-hot encode
    h3_dummies = pd.get_dummies(df[h3_col], prefix=f"{h3_col}_enc", dtype=int)
    
    logger.info(f"Created {len(h3_dummies.columns)} one-hot features for {h3_col}")
    
    # Concatenate
    df = pd.concat([df, h3_dummies], axis=1)
    
    return df


def feature_engineering_pipeline(
    artifacts_dir: Path, neighbors: int = 10
) -> None:
    """
    Full feature engineering pipeline.
    
    Parameters
    ----------
    artifacts_dir : Path
        Directory with train.parquet and where to save outputs
    neighbors : int
        Number of neighbors for KNN graph
    """
    logger.info("Loading train data...")
    train_df = pd.read_parquet(artifacts_dir / "train.parquet")
    
    logger.info(f"Train data: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
    
    # Add H3 features (resolution 8 for features, resolution 7 for CV blocks)
    logger.info("\nAdding H3 spatial features...")
    train_df = add_h3_features(train_df, resolution=8)
    train_df = add_h3_features(train_df, resolution=7)  # For CV blocks
    
    # Add geocenter distance
    logger.info("\nAdding distance to geocenter...")
    train_df, center_lat, center_lon = add_geocenter_distance(train_df)
    
    # Save geocenter for test set
    geocenter = {"center_lat": center_lat, "center_lon": center_lon}
    with open(artifacts_dir / "geocenter.pkl", "wb") as f:
        pickle.dump(geocenter, f)
    logger.info(f"Saved: {artifacts_dir / 'geocenter.pkl'}")
    
    # Build neighbor graph
    logger.info(f"\nBuilding KNN neighbor graph (k={neighbors})...")
    neighbor_indices, neighbor_distances = compute_neighbor_graph(train_df, k=neighbors)
    
    # Save neighbor graph
    neighbor_graph = {
        "indices": neighbor_indices,
        "distances": neighbor_distances,
    }
    
    with open(artifacts_dir / "neighbor_graph.pkl", "wb") as f:
        pickle.dump(neighbor_graph, f)
    logger.info(f"Saved: {artifacts_dir / 'neighbor_graph.pkl'}")
    
    # Add neighbor density features
    logger.info("\nAdding neighbor density features...")
    train_df = add_neighbor_density_features(train_df, neighbor_distances)
    
    # Add neighbor aggregate features
    logger.info("\nAdding neighbor aggregate features...")
    train_df = add_neighbor_aggregate_features(train_df, neighbor_distances)
    
    # Add cyclical time features
    logger.info("\nAdding cyclical time features...")
    train_df = add_cyclical_time_features(train_df)
    
    # Add one-hot encoding for H3 (for XGBoost)
    logger.info("\nAdding one-hot encoding for h3_r8...")
    train_df = add_h3_onehot(train_df, h3_col="h3_r8")
    
    # Save H3 categories for test set
    h3_categories = {"h3_r8": train_df["h3_r8"].unique().tolist()}
    with open(artifacts_dir / "h3_categories.pkl", "wb") as f:
        pickle.dump(h3_categories, f)
    logger.info(f"Saved: {artifacts_dir / 'h3_categories.pkl'}")
    
    # Save feature-engineered train data
    train_df.to_parquet(artifacts_dir / "features.parquet", index=False)
    logger.info(f"\nSaved: {artifacts_dir / 'features.parquet'}")
    
    logger.info(f"Final shape: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
    logger.info("\nFeature engineering complete!")


def apply_features_to_test(artifacts_dir: Path) -> None:
    """
    Apply same features to test set.
    
    Parameters
    ----------
    artifacts_dir : Path
        Directory with test.parquet and neighbor_graph.pkl
    """
    logger.info("Loading test data and neighbor graph...")
    test_df = pd.read_parquet(artifacts_dir / "test.parquet")
    
    with open(artifacts_dir / "neighbor_graph.pkl", "rb") as f:
        neighbor_graph = pickle.load(f)
    
    neighbor_distances = neighbor_graph["distances"]
    
    # Load H3 categories from train
    with open(artifacts_dir / "h3_categories.pkl", "rb") as f:
        h3_categories = pickle.load(f)
    
    # Load geocenter from train
    with open(artifacts_dir / "geocenter.pkl", "rb") as f:
        geocenter = pickle.load(f)
    
    logger.info(f"Test data: {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")
    
    # Apply same transformations
    logger.info("\nApplying features to test set...")
    test_df = add_h3_features(test_df, resolution=8)
    test_df = add_h3_features(test_df, resolution=7)
    test_df, _, _ = add_geocenter_distance(test_df, geocenter["center_lat"], geocenter["center_lon"])
    test_df = add_neighbor_density_features(test_df, neighbor_distances)
    test_df = add_neighbor_aggregate_features(test_df, neighbor_distances)
    test_df = add_cyclical_time_features(test_df)
    
    # Add one-hot encoding for H3 (aligned with train)
    logger.info("\nAdding one-hot encoding for h3_r8 (aligned with train)...")
    test_df = add_h3_onehot(test_df, h3_col="h3_r8")
    
    # Align columns with train (add missing, drop extra)
    train_df = pd.read_parquet(artifacts_dir / "features.parquet")
    train_h3_cols = [col for col in train_df.columns if col.startswith("h3_r8_enc_")]
    test_h3_cols = [col for col in test_df.columns if col.startswith("h3_r8_enc_")]
    
    # Add missing columns (with 0s)
    for col in train_h3_cols:
        if col not in test_h3_cols:
            test_df[col] = 0
    
    logger.info(f"Aligned H3 one-hot columns: {len(train_h3_cols)} features")
    
    # Save
    test_df.to_parquet(artifacts_dir / "features_test.parquet", index=False)
    logger.info(f"Saved: {artifacts_dir / 'features_test.parquet'}")
    logger.info(f"Final shape: {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")

