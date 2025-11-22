"""Model training: XGBoost OOF baseline and CatBoost with neighbor lags."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from src.spatial_cv import load_folds
from src.utils import compute_metrics, print_metrics
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Feature columns for modeling
BASE_NUMERIC_FEATURES = [
    "hour",
    "dow",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "year_day_sin",
    "year_day_cos",
    "dist_to_center",
    "neighbor_mean_dist",
    "neighbor_median_dist",
    # Weather features (will be added)
    "temperature",
    "wind_speed",
    "rain",
    "snowfall",
    # Holiday feature
    "is_holiday",
    # Lagged features
    "count_lag_24h",
    "count_rolling_mean_7d",
]

# XGBoost uses one-hot encoded h3_r8 (will be added dynamically)
# CatBoost uses original h3_r8 string (handles categorical natively)
FEATURE_COLS_CATBOOST_BASE = BASE_NUMERIC_FEATURES + ["h3_r8"]
FEATURE_COLS_CATBOOST_WITH_LAG = BASE_NUMERIC_FEATURES + ["h3_r8", "spatial_oof_neighbors_mean"]

CATEGORICAL_FEATURES = ["h3_r8", "dow", "month"]


def get_xgboost_features(df: pd.DataFrame) -> list:
    """Get feature list for XGBoost including one-hot encoded h3_r8."""
    h3_onehot_cols = [col for col in df.columns if col.startswith("h3_r8_enc_")]
    return BASE_NUMERIC_FEATURES + h3_onehot_cols


def get_xgboost_features_with_lag(df: pd.DataFrame) -> list:
    """Get feature list for XGBoost with lag including one-hot encoded h3_r8."""
    h3_onehot_cols = [col for col in df.columns if col.startswith("h3_r8_enc_")]
    return BASE_NUMERIC_FEATURES + h3_onehot_cols + ["spatial_oof_neighbors_mean"]


def _make_neighbor_lag_computer(
    oof_matrix: pd.DataFrame,
    neighbor_indices: Dict[str, np.ndarray],
):
    """
    Create a closure function for computing neighbor lag from OOF matrix.
    
    This function is extracted to avoid duplication across:
    - create_neighbor_oof_lags()
    - rolling_window_validation()
    
    Parameters
    ----------
    oof_matrix : DataFrame
        Pivot table of OOF predictions (timestamp x site_id)
    neighbor_indices : dict
        {site_id: array of neighbor site_ids}
        
    Returns
    -------
    function
        Function that takes a row and returns neighbor mean OOF
    """
    def compute_neighbor_mean(row):
        """Compute mean of neighbors' OOF predictions at same timestamp."""
        site_id = row["site_id"]
        timestamp = row["timestamp"]
        neighbors = neighbor_indices.get(site_id, [])
        
        if len(neighbors) > 0 and timestamp in oof_matrix.index:
            valid_neighbors = [n for n in neighbors if n in oof_matrix.columns]
            if valid_neighbors:
                neighbor_vals = oof_matrix.loc[timestamp, valid_neighbors]
                if neighbor_vals.notna().any():
                    return neighbor_vals.mean()
        return np.nan
    
    return compute_neighbor_mean


def _make_test_neighbor_lag_computer(
    train_df: pd.DataFrame,
    neighbor_indices: Dict[str, np.ndarray],
    oof_col: str = "oof_pred"
):
    """
    Create a closure function for computing test neighbor lag from train aggregates.
    
    This function is extracted to avoid duplication across:
    - train_final_and_predict_test()
    - rolling_window_validation()
    
    Parameters
    ----------
    train_df : DataFrame
        Training data with OOF predictions
    neighbor_indices : dict
        {site_id: array of neighbor site_ids}
    oof_col : str
        Name of OOF prediction column
        
    Returns
    -------
    function
        Function that takes a row and returns neighbor lag for test
    """
    # Pre-compute aggregations
    train_site_hour_dow = train_df.groupby(["site_id", "hour", "dow"])[oof_col].mean().to_dict()
    train_site_hour = train_df.groupby(["site_id", "hour"])[oof_col].mean().to_dict()
    train_h3_hour = train_df.groupby(["h3_r7", "hour"])[oof_col].median().to_dict()
    global_median = train_df[oof_col].mean()
    
    def compute_test_neighbor_lag(row):
        """Compute neighbor lag for test row using train OOF."""
        site_id = row["site_id"]
        hour = row["hour"]
        dow = row["dow"]
        h3_block = row["h3_r7"]
        neighbors = neighbor_indices.get(site_id, np.array([]))
        
        if len(neighbors) == 0:
            return train_h3_hour.get((h3_block, hour), global_median)
        
        # Try hour+dow
        neighbor_vals = []
        for neighbor in neighbors:
            key = (neighbor, hour, dow)
            if key in train_site_hour_dow:
                neighbor_vals.append(train_site_hour_dow[key])
        
        if neighbor_vals:
            return np.mean(neighbor_vals)
        
        # Fallback: hour only
        for neighbor in neighbors:
            key = (neighbor, hour)
            if key in train_site_hour:
                neighbor_vals.append(train_site_hour[key])
        
        if neighbor_vals:
            return np.mean(neighbor_vals)
        
        # Final fallback: block median or global
        return train_h3_hour.get((h3_block, hour), global_median)
    
    return compute_test_neighbor_lag


def add_lagged_features_no_leakage(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str = "count"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add lagged features without data leakage.
    
    Computes lags on concatenated train+valid data (sorted by time),
    ensuring that validation lags only use past data from train.
    
    Parameters
    ----------
    train_df : DataFrame
        Training data
    valid_df : DataFrame
        Validation data
    target_col : str
        Target column name
        
    Returns
    -------
    train_with_lags : DataFrame
        Train data with lagged features (same order as input)
    valid_with_lags : DataFrame
        Valid data with lagged features (same order as input)
    """
    from src.advanced_features import add_lagged_features

    # Concatenate train and valid (in temporal order)
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    
    # Save original order
    train_df['_original_order'] = range(len(train_df))
    valid_df['_original_order'] = range(len(valid_df))
    
    # Mark to split later
    train_df['_split'] = 'train'
    valid_df['_split'] = 'valid'
    
    combined = pd.concat([train_df, valid_df], ignore_index=True)
    combined = combined.sort_values(['site_id', 'timestamp'])
    
    # Compute lags on combined data
    combined = add_lagged_features(combined, target_col)
    
    # Split back and restore original order
    train_with_lags = (combined[combined['_split'] == 'train']
                       .sort_values('_original_order')
                       .drop(columns=['_split', '_original_order'])
                       .reset_index(drop=True))
    
    valid_with_lags = (combined[combined['_split'] == 'valid']
                       .sort_values('_original_order')
                       .drop(columns=['_split', '_original_order'])
                       .reset_index(drop=True))
    
    return train_with_lags, valid_with_lags


def prepare_xy(
    df: pd.DataFrame, feature_cols: List[str], target_col: str = "count"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare X and y for modeling.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    feature_cols : list of str
        Feature column names
    target_col : str
        Target column name
        
    Returns
    -------
    X : DataFrame
        Features
    y : ndarray
        Target
    """
    X = df[feature_cols].copy()
    y = df[target_col].values
    return X, y


def train_xgboost_oof(
    df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    feature_cols_base: List[str],
    target_col: str = "count",
) -> Tuple[np.ndarray, List[Dict], List]:
    """
    Train XGBoost with OOF predictions.
    
    IMPORTANT: Computes lagged features INSIDE each fold to prevent data leakage.
    
    Parameters
    ----------
    df : DataFrame
        Training dataframe WITHOUT lagged features
    folds : list of tuples
        (train_indices, valid_indices) for each fold
    feature_cols_base : list of str
        Base feature columns (lagged features will be added)
    target_col : str
        Target column
        
    Returns
    -------
    oof_preds : ndarray
        Out-of-fold predictions for entire dataset
    fold_metrics : list of dicts
        Metrics for each fold
    models : list
        Trained XGBoost models for each fold
    """
    logger.info("=" * 60)
    logger.info("Stage 1: XGBoost OOF Baseline")
    logger.info("=" * 60)
    
    # Initialize OOF predictions
    oof_preds = np.zeros(len(df))
    fold_metrics = []
    models = []
    
    # XGBoost parameters
    params = {
        "max_depth": 6,
        "n_estimators": 800,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    
    logger.info(f"XGBoost parameters: {params}")
    logger.info(f"Base features (before lags): {len(feature_cols_base)}")
    logger.info("Computing lagged features inside each fold to prevent leakage...")
    
    # Train on each fold
    for fold_idx, (train_idx, valid_idx) in enumerate(tqdm(folds, desc="XGBoost folds")):
        # Get fold data WITHOUT lags
        fold_train = df.iloc[train_idx].copy()
        fold_valid = df.iloc[valid_idx].copy()
        
        # Add lagged features without leakage
        fold_train, fold_valid = add_lagged_features_no_leakage(
            fold_train, fold_valid, target_col
        )
        
        # Prepare X, y
        X_train, y_train = prepare_xy(fold_train, feature_cols_base, target_col)
        X_valid, y_valid = prepare_xy(fold_valid, feature_cols_base, target_col)
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        
        # Predict on validation
        preds_valid = model.predict(X_valid)
        oof_preds[valid_idx] = preds_valid
        
        # Compute metrics
        metrics = compute_metrics(y_valid, preds_valid)
        fold_metrics.append(metrics)
        
        logger.info(
            f"Fold {fold_idx + 1}/{len(folds)}: "
            f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
            f"MAPE={metrics['mape']:.2f}%, WAPE={metrics['wape']:.2f}%"
        )
        
        models.append(model)
    
    # Overall OOF metrics
    y_all = df[target_col].values
    overall_metrics = compute_metrics(y_all, oof_preds)
    logger.info("\n" + "=" * 60)
    logger.info("XGBoost OOF Performance (all folds):")
    print_metrics(overall_metrics)
    logger.info("=" * 60 + "\n")
    
    return oof_preds, fold_metrics, models


def create_neighbor_oof_lags(
    df: pd.DataFrame,
    oof_preds: np.ndarray,
    neighbor_indices: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Create spatial neighbor OOF lag feature (VECTORIZED).
    
    For each row, compute mean of OOF predictions from K nearest neighbors
    at the same timestamp.
    
    Parameters
    ----------
    df : DataFrame
        Training dataframe with site_id, timestamp, h3_r7
    oof_preds : ndarray
        OOF predictions from Stage 1
    neighbor_indices : dict
        {site_id: array of neighbor site_ids}
        
    Returns
    -------
    df : DataFrame
        DataFrame with added spatial_oof_neighbors_mean column
    """
    logger.info("Creating neighbor OOF lag features (vectorized)...")
    
    df = df.copy()
    df["oof_pred"] = oof_preds
    
    # Create OOF matrix: pivot table (timestamp x site_id)
    logger.info("Building OOF pivot table (timestamp x site_id)...")
    oof_matrix = df.pivot_table(
        index="timestamp",
        columns="site_id",
        values="oof_pred",
        aggfunc="mean"  # Handle duplicates if any
    )
    
    logger.info(f"OOF matrix shape: {oof_matrix.shape}")
    
    # Vectorized computation of neighbor means
    logger.info("Computing neighbor means (vectorized)...")
    compute_neighbor_mean = _make_neighbor_lag_computer(oof_matrix, neighbor_indices)
    df["spatial_oof_neighbors_mean"] = df.apply(compute_neighbor_mean, axis=1)
    
    # Fill NaNs with h3 block median at same timestamp
    logger.info("Filling missing values with block medians...")
    h3_timestamp_medians = df.groupby(["h3_r7", "timestamp"])["oof_pred"].transform("median")
    df["spatial_oof_neighbors_mean"] = df["spatial_oof_neighbors_mean"].fillna(h3_timestamp_medians)
    
    # Final fallback: global median
    global_median = df["oof_pred"].median()
    df["spatial_oof_neighbors_mean"] = df["spatial_oof_neighbors_mean"].fillna(global_median)
    
    logger.info(
        f"Neighbor OOF lag created. Mean: {df['spatial_oof_neighbors_mean'].mean():.2f}, "
        f"Std: {df['spatial_oof_neighbors_mean'].std():.2f}"
    )
    
    # Drop temporary column
    df = df.drop(columns=["oof_pred"])
    
    return df


def train_catboost_oof(
    df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    feature_cols: List[str],
    categorical_features: List[str],
    target_col: str = "count",
) -> Tuple[np.ndarray, List[Dict], List]:
    """
    Train CatBoost with neighbor OOF lag.
    
    IMPORTANT: Computes lagged features INSIDE each fold to prevent data leakage.
    
    Parameters
    ----------
    df : DataFrame
        Training dataframe with spatial_oof_neighbors_mean but WITHOUT lagged features
    folds : list of tuples
        Same folds as Stage 1
    feature_cols : list of str
        Feature columns (including spatial_oof_neighbors_mean)
    categorical_features : list of str
        Categorical feature names
    target_col : str
        Target column
        
    Returns
    -------
    oof_preds : ndarray
        Out-of-fold predictions
    fold_metrics : list of dicts
        Metrics for each fold
    models : list
        Trained CatBoost models
    """
    logger.info("=" * 60)
    logger.info("Stage 2: CatBoost with Neighbor OOF Lag")
    logger.info("=" * 60)
    logger.info("Computing lagged features inside each fold to prevent leakage...")
    
    # Get indices of categorical features
    cat_feature_indices = [i for i, col in enumerate(feature_cols) if col in categorical_features]
    
    # Initialize OOF predictions
    oof_preds = np.zeros(len(df))
    fold_metrics = []
    models = []
    
    # CatBoost parameters
    params = {
        "depth": 8,
        "iterations": 1500,
        "learning_rate": 0.03,
        "l2_leaf_reg": 6,
        "loss_function": "RMSE",
        "bootstrap_type": "Bayesian",
        "random_strength": 1.5,
        "early_stopping_rounds": 100,
        "random_state": 42,
        "verbose": False,
        "thread_count": -1,
    }
    
    logger.info(f"CatBoost parameters: {params}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Categorical features: {categorical_features}")
    
    # Train on each fold
    for fold_idx, (train_idx, valid_idx) in enumerate(tqdm(folds, desc="CatBoost folds")):
        # Get fold data WITHOUT lags (but WITH spatial_oof_neighbors_mean already present)
        fold_train = df.iloc[train_idx].copy()
        fold_valid = df.iloc[valid_idx].copy()
        
        # Add lagged features without leakage
        fold_train, fold_valid = add_lagged_features_no_leakage(
            fold_train, fold_valid, target_col
        )
        
        # Prepare X, y
        X_train, y_train = prepare_xy(fold_train, feature_cols, target_col)
        X_valid, y_valid = prepare_xy(fold_valid, feature_cols, target_col)
        
        # Create CatBoost pools
        train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_feature_indices)
        
        # Train model
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        
        # Predict on validation
        preds_valid = model.predict(X_valid)
        oof_preds[valid_idx] = preds_valid
        
        # Compute metrics
        metrics = compute_metrics(y_valid, preds_valid)
        fold_metrics.append(metrics)
        
        logger.info(
            f"Fold {fold_idx + 1}/{len(folds)}: "
            f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
            f"MAPE={metrics['mape']:.2f}%, WAPE={metrics['wape']:.2f}%"
        )
        
        models.append(model)
    
    # Overall OOF metrics
    y_all = df[target_col].values
    overall_metrics = compute_metrics(y_all, oof_preds)
    logger.info("\n" + "=" * 60)
    logger.info("CatBoost OOF Performance (all folds):")
    print_metrics(overall_metrics)
    logger.info("=" * 60 + "\n")
    
    return oof_preds, fold_metrics, models


def train_final_and_predict_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_oof_preds: np.ndarray,
    neighbor_indices: Dict[str, np.ndarray],
    feature_cols: List[str],
    categorical_features: List[str],
    target_col: str = "count",
) -> np.ndarray:
    """
    Train final model on all TRAIN data and predict TEST.
    
    Parameters
    ----------
    train_df : DataFrame
        Full training dataframe with neighbor OOF lag
    test_df : DataFrame
        Test dataframe
    train_oof_preds : ndarray
        OOF predictions from training (for creating test neighbor lags)
    neighbor_indices : dict
        Neighbor graph
    feature_cols : list of str
        Feature columns
    categorical_features : list of str
        Categorical features
    target_col : str
        Target column
        
    Returns
    -------
    test_preds : ndarray
        Predictions on test set
    """
    logger.info("=" * 60)
    logger.info("Final Model: Train on full TRAIN, predict TEST")
    logger.info("=" * 60)
    
    # Create neighbor lag for test set (VECTORIZED)
    # For test, use train OOF predictions to create neighbor lags
    logger.info("Creating neighbor lags for test set from train OOF (vectorized)...")
    
    test_df = test_df.copy()
    train_df_with_oof = train_df.copy()
    train_df_with_oof["oof_pred"] = train_oof_preds
    
    # Use extracted function for consistency
    compute_test_neighbor_lag = _make_test_neighbor_lag_computer(
        train_df_with_oof, neighbor_indices, oof_col="oof_pred"
    )
    test_df["spatial_oof_neighbors_mean"] = test_df.apply(compute_test_neighbor_lag, axis=1)
    test_df["spatial_oof_neighbors_mean"] = test_df["spatial_oof_neighbors_mean"].fillna(train_oof_preds.mean())
    
    logger.info(
        f"Test neighbor lag created. Mean: {test_df['spatial_oof_neighbors_mean'].mean():.2f}"
    )
    
    # Add lagged features without leakage: concat train+test, compute lags, split
    logger.info("Adding lagged features (train+test concat to prevent leakage)...")
    train_df, test_df = add_lagged_features_no_leakage(train_df, test_df, target_col)
    
    # Prepare data
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)
    X_test, y_test = prepare_xy(test_df, feature_cols, target_col)
    
    cat_feature_indices = [i for i, col in enumerate(feature_cols) if col in categorical_features]
    
    # Train final CatBoost
    params = {
        "depth": 8,
        "iterations": 1500,
        "learning_rate": 0.03,
        "l2_leaf_reg": 6,
        "loss_function": "RMSE",
        "bootstrap_type": "Bayesian",
        "random_strength": 1.5,
        "random_state": 42,
        "verbose": 100,
        "thread_count": -1,
    }
    
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    
    logger.info("Training final CatBoost on full TRAIN data...")
    model = CatBoostRegressor(**params)
    model.fit(train_pool)
    
    # Predict test
    logger.info("Predicting on TEST set...")
    test_preds = model.predict(X_test)
    
    # Compute test metrics
    test_metrics = compute_metrics(y_test, test_preds)
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET Performance:")
    print_metrics(test_metrics)
    logger.info("=" * 60 + "\n")
    
    return test_preds, model


def models_pipeline(artifacts_dir: Path) -> None:
    """
    Full modeling pipeline: XGBoost OOF -> CatBoost with neighbor lag -> Test predictions.
    
    Parameters
    ----------
    artifacts_dir : Path
        Directory with features.parquet, features_test.parquet, folds.pkl, neighbor_graph.pkl
        
    Note
    ----
    Number of folds and neighbors are loaded from saved artifacts (folds.pkl, neighbor_graph.pkl)
    created by spatial_cv_pipeline and feature_engineering_pipeline respectively.
    """
    logger.info("Loading data...")
    train_df = pd.read_parquet(artifacts_dir / "features.parquet")
    test_df = pd.read_parquet(artifacts_dir / "features_test.parquet")
    folds = load_folds(artifacts_dir / "folds.pkl")
    
    with open(artifacts_dir / "neighbor_graph.pkl", "rb") as f:
        neighbor_graph = pickle.load(f)
    neighbor_indices = neighbor_graph["indices"]
    
    logger.info(f"Train: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
    logger.info(f"Test:  {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")
    logger.info(f"Folds: {len(folds)}")
    
    # Remove lagged features if present (will be computed per-fold to prevent leakage)
    lag_cols = ["count_lag_24h", "count_rolling_mean_7d"]
    for col in lag_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
            logger.info(f"Removed {col} (will be computed per-fold)")
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
    
    # Get XGBoost features (including one-hot h3_r8, will add lags per-fold)
    xgb_features = get_xgboost_features(train_df)
    logger.info(f"XGBoost base features: {len(xgb_features)} (including {len([c for c in xgb_features if 'h3_r8_enc' in c])} H3 one-hot)")
    logger.info("Lagged features will be computed inside each fold to prevent data leakage")
    
    # Stage 1: XGBoost OOF (will compute lags per-fold)
    xgb_oof_preds, xgb_fold_metrics, xgb_models = train_xgboost_oof(
        train_df, folds, xgb_features, target_col="count"
    )
    
    # Create neighbor OOF lags
    train_df = create_neighbor_oof_lags(train_df, xgb_oof_preds, neighbor_indices)
    
    # Stage 2: CatBoost with neighbor lag (includes h3_r8)
    cb_oof_preds, cb_fold_metrics, cb_models = train_catboost_oof(
        train_df, folds, FEATURE_COLS_CATBOOST_WITH_LAG, CATEGORICAL_FEATURES, target_col="count"
    )
    
    # Final: Train on full TRAIN and predict TEST (use CatBoost features)
    test_preds, final_model = train_final_and_predict_test(
        train_df,
        test_df,
        xgb_oof_preds,
        neighbor_indices,
        FEATURE_COLS_CATBOOST_WITH_LAG,
        CATEGORICAL_FEATURES,
        target_col="count",
    )
    
    # Save predictions
    test_df["predicted_count"] = test_preds
    test_df.to_parquet(artifacts_dir / "test_predictions.parquet", index=False)
    logger.info(f"Saved: {artifacts_dir / 'test_predictions.parquet'}")
    logger.info(f"Test predictions shape: {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")
    logger.info(f"Unique sites in test predictions: {test_df['site_id'].nunique()}")
    
    # Save models and metrics
    results = {
        "xgb_fold_metrics": xgb_fold_metrics,
        "cb_fold_metrics": cb_fold_metrics,
        "xgb_oof_preds": xgb_oof_preds,
        "cb_oof_preds": cb_oof_preds,
        "final_model": final_model,
    }
    
    with open(artifacts_dir / "model_results.pkl", "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved: {artifacts_dir / 'model_results.pkl'}")
    
    # Save metrics summary as CSV
    logger.info("\nSaving metrics summary...")
    xgb_metrics_df = pd.DataFrame(xgb_fold_metrics)
    xgb_metrics_df["model"] = "XGBoost"
    xgb_metrics_df["fold"] = range(1, len(xgb_fold_metrics) + 1)
    
    cb_metrics_df = pd.DataFrame(cb_fold_metrics)
    cb_metrics_df["model"] = "CatBoost"
    cb_metrics_df["fold"] = range(1, len(cb_fold_metrics) + 1)
    
    all_metrics = pd.concat([xgb_metrics_df, cb_metrics_df], ignore_index=True)
    all_metrics.to_csv(artifacts_dir / "cv_metrics.csv", index=False)
    logger.info(f"Saved: {artifacts_dir / 'cv_metrics.csv'}")
    
    # Compute mean metrics per model
    mean_metrics = all_metrics.groupby("model")[["rmse", "mae", "mape", "wape"]].mean()
    logger.info("\n=== MEAN CV METRICS (10 folds) ===")
    logger.info(mean_metrics.to_string())
    
    logger.info("\nModeling pipeline complete!")


def rolling_window_validation(
    artifacts_dir: Path,
    train_years: int = 2,
    test_years: int = 1,
) -> Dict:
    """
    Rolling window validation: train on N years -> test on M years.
    
    Iterates through all data with sliding window:
    - Window 1: 2005-2006 train → 2007 test
    - Window 2: 2006-2007 train → 2008 test
    - ...
    - Window K: years [i:i+train_years] train → years [i+train_years:i+train_years+test_years] test
    
    Parameters
    ----------
    artifacts_dir : Path
        Directory with clean.parquet (full dataset)
    train_years : int
        Number of years for training window (default 2)
    test_years : int
        Number of years for test window (default 1)
        
    Returns
    -------
    dict
        Results dictionary with metrics per window and averaged metrics
    """
    logger.info("=" * 60)
    logger.info("Rolling Window Validation")
    logger.info("=" * 60)
    logger.info(f"Train window: {train_years} years")
    logger.info(f"Test window: {test_years} years")
    logger.info("")
    
    # Load full dataset
    df = pd.read_parquet(artifacts_dir / "clean.parquet")
    logger.info(f"Loaded full dataset: {len(df):,} rows")
    
    # Extract year
    df["year"] = df["timestamp"].dt.year
    years = sorted(df["year"].unique())
    logger.info(f"Available years: {years[0]}-{years[-1]} ({len(years)} years)")
    
    # Determine windows
    n_windows = len(years) - train_years - test_years + 1
    if n_windows < 1:
        raise ValueError(
            f"Not enough years for rolling window validation. "
            f"Need at least {train_years + test_years} years, but have {len(years)}."
        )
    
    logger.info(f"Number of windows: {n_windows}")
    logger.info("")
    
    # Load neighbor graph (built on full data)
    with open(artifacts_dir / "neighbor_graph.pkl", "rb") as f:
        neighbor_graph = pickle.load(f)
    neighbor_indices = neighbor_graph["indices"]
    
    # Results storage
    windows_results = []
    
    # Iterate through windows
    for window_idx in range(n_windows):
        train_start_year = years[window_idx]
        train_end_year = years[window_idx + train_years - 1]
        test_start_year = years[window_idx + train_years]
        test_end_year = years[window_idx + train_years + test_years - 1]
        
        logger.info("=" * 60)
        logger.info(f"Window {window_idx + 1}/{n_windows}")
        logger.info(f"  Train: {train_start_year}-{train_end_year}")
        logger.info(f"  Test:  {test_start_year}-{test_end_year}")
        logger.info("=" * 60)
        
        # Split data
        train_mask = (df["year"] >= train_start_year) & (df["year"] <= train_end_year)
        test_mask = (df["year"] >= test_start_year) & (df["year"] <= test_end_year)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        logger.info(f"Train: {len(train_df):,} rows, {train_df['site_id'].nunique()} sites")
        logger.info(f"Test:  {len(test_df):,} rows, {test_df['site_id'].nunique()} sites")
        
        # Apply feature engineering
        logger.info("\nApplying feature engineering...")
        from src.advanced_features import (add_holiday_features,
                                           add_lagged_features,
                                           add_weather_features)
        from src.features import (add_cyclical_time_features,
                                  add_geocenter_distance, add_h3_features,
                                  add_h3_onehot,
                                  add_neighbor_aggregate_features,
                                  add_neighbor_density_features)

        # Load neighbor distances
        neighbor_distances = neighbor_graph["distances"]
        
        # Add features to train
        train_df = add_h3_features(train_df, resolution=8)
        train_df = add_h3_features(train_df, resolution=7)
        train_df, center_lat, center_lon = add_geocenter_distance(train_df)
        train_df = add_neighbor_density_features(train_df, neighbor_distances)
        train_df = add_neighbor_aggregate_features(train_df, neighbor_distances)
        train_df = add_cyclical_time_features(train_df)
        train_df = add_h3_onehot(train_df, h3_col="h3_r8")
        
        # Add features to test (using train's geocenter)
        test_df = add_h3_features(test_df, resolution=8)
        test_df = add_h3_features(test_df, resolution=7)
        test_df, _, _ = add_geocenter_distance(test_df, center_lat, center_lon)
        test_df = add_neighbor_density_features(test_df, neighbor_distances)
        test_df = add_neighbor_aggregate_features(test_df, neighbor_distances)
        test_df = add_cyclical_time_features(test_df)
        test_df = add_h3_onehot(test_df, h3_col="h3_r8")
        
        # Align H3 one-hot columns between train and test
        train_h3_cols = [col for col in train_df.columns if col.startswith("h3_r8_enc_")]
        test_h3_cols = [col for col in test_df.columns if col.startswith("h3_r8_enc_")]
        for col in train_h3_cols:
            if col not in test_h3_cols:
                test_df[col] = 0
        for col in test_h3_cols:
            if col not in train_h3_cols:
                train_df[col] = 0
        
        # Add advanced features
        logger.info("\nAdding weather features...")
        weather_path = artifacts_dir.parent / "data" / "open-meteo-55.71N12.44E6m.csv"
        if weather_path.exists():
            train_df = add_weather_features(train_df, weather_path, location_id=0)
            test_df = add_weather_features(test_df, weather_path, location_id=0)
        else:
            logger.warning(f"Weather file not found: {weather_path}, skipping weather features")
        
        logger.info("Adding holiday features...")
        train_df = add_holiday_features(train_df)
        test_df = add_holiday_features(test_df)
        
        logger.info("Adding lagged features (train+test concat to prevent leakage)...")
        train_df, test_df = add_lagged_features_no_leakage(train_df, test_df, "count")
        
        # Get features
        xgb_features = get_xgboost_features(train_df)
        
        # Train XGBoost
        logger.info("\nTraining XGBoost...")
        X_train_xgb, y_train = prepare_xy(train_df, xgb_features, "count")
        X_test_xgb, y_test = prepare_xy(test_df, xgb_features, "count")
        
        xgb_params = {
            "max_depth": 6,
            "n_estimators": 800,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_xgb, y_train, verbose=False)
        
        # Predictions
        train_preds_xgb = xgb_model.predict(X_train_xgb)
        test_preds_xgb = xgb_model.predict(X_test_xgb)
        
        # XGBoost metrics
        xgb_metrics = compute_metrics(y_test, test_preds_xgb)
        logger.info(f"XGBoost Test: RMSE={xgb_metrics['rmse']:.2f}, MAE={xgb_metrics['mae']:.2f}, "
                   f"WAPE={xgb_metrics['wape']:.2f}%")
        
        # Create neighbor lags for CatBoost
        logger.info("\nCreating neighbor lags for CatBoost...")
        train_df["oof_pred"] = train_preds_xgb
        
        # Build OOF matrix for train
        oof_matrix = train_df.pivot_table(
            index="timestamp", columns="site_id", values="oof_pred", aggfunc="mean"
        )
        
        # Use extracted functions for consistency
        compute_neighbor_mean = _make_neighbor_lag_computer(oof_matrix, neighbor_indices)
        train_df["spatial_oof_neighbors_mean"] = train_df.apply(compute_neighbor_mean, axis=1)
        h3_timestamp_medians = train_df.groupby(["h3_r7", "timestamp"])["oof_pred"].transform("median")
        train_df["spatial_oof_neighbors_mean"] = train_df["spatial_oof_neighbors_mean"].fillna(h3_timestamp_medians)
        train_df["spatial_oof_neighbors_mean"] = train_df["spatial_oof_neighbors_mean"].fillna(train_df["oof_pred"].median())
        
        # Create neighbor lags for test
        compute_test_neighbor_lag = _make_test_neighbor_lag_computer(
            train_df, neighbor_indices, oof_col="oof_pred"
        )
        test_df["spatial_oof_neighbors_mean"] = test_df.apply(compute_test_neighbor_lag, axis=1)
        test_df["spatial_oof_neighbors_mean"] = test_df["spatial_oof_neighbors_mean"].fillna(train_df["oof_pred"].mean())
        
        # Train CatBoost
        logger.info("\nTraining CatBoost...")
        X_train_cb, _ = prepare_xy(train_df, FEATURE_COLS_CATBOOST_WITH_LAG, "count")
        X_test_cb, _ = prepare_xy(test_df, FEATURE_COLS_CATBOOST_WITH_LAG, "count")
        
        cat_feature_indices = [i for i, col in enumerate(FEATURE_COLS_CATBOOST_WITH_LAG) 
                              if col in CATEGORICAL_FEATURES]
        
        cb_params = {
            "depth": 8,
            "iterations": 1500,
            "learning_rate": 0.03,
            "l2_leaf_reg": 6,
            "loss_function": "RMSE",
            "bootstrap_type": "Bayesian",
            "random_strength": 1.5,
            "random_state": 42,
            "verbose": False,
            "thread_count": -1,
        }
        
        train_pool = Pool(X_train_cb, y_train, cat_features=cat_feature_indices)
        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(train_pool)
        
        # Predictions
        test_preds_cb = cb_model.predict(X_test_cb)
        
        # CatBoost metrics
        cb_metrics = compute_metrics(y_test, test_preds_cb)
        logger.info(f"CatBoost Test: RMSE={cb_metrics['rmse']:.2f}, MAE={cb_metrics['mae']:.2f}, "
                   f"WAPE={cb_metrics['wape']:.2f}%")
        
        # Store results
        windows_results.append({
            "window": window_idx + 1,
            "train_years": f"{train_start_year}-{train_end_year}",
            "test_years": f"{test_start_year}-{test_end_year}",
            "xgb_rmse": xgb_metrics["rmse"],
            "xgb_mae": xgb_metrics["mae"],
            "xgb_wape": xgb_metrics["wape"],
            "cb_rmse": cb_metrics["rmse"],
            "cb_mae": cb_metrics["mae"],
            "cb_wape": cb_metrics["wape"],
        })
        
        logger.info("")
    
    # Compute averaged metrics
    logger.info("=" * 60)
    logger.info("Rolling Window Validation - Summary")
    logger.info("=" * 60)
    
    results_df = pd.DataFrame(windows_results)
    
    # Average metrics
    avg_xgb_rmse = results_df["xgb_rmse"].mean()
    avg_xgb_mae = results_df["xgb_mae"].mean()
    avg_xgb_wape = results_df["xgb_wape"].mean()
    
    avg_cb_rmse = results_df["cb_rmse"].mean()
    avg_cb_mae = results_df["cb_mae"].mean()
    avg_cb_wape = results_df["cb_wape"].mean()
    
    logger.info(f"\nXGBoost (averaged over {n_windows} windows):")
    logger.info(f"  RMSE: {avg_xgb_rmse:.2f} (±{results_df['xgb_rmse'].std():.2f})")
    logger.info(f"  MAE:  {avg_xgb_mae:.2f} (±{results_df['xgb_mae'].std():.2f})")
    logger.info(f"  WAPE: {avg_xgb_wape:.2f}% (±{results_df['xgb_wape'].std():.2f}%)")
    
    logger.info(f"\nCatBoost (averaged over {n_windows} windows):")
    logger.info(f"  RMSE: {avg_cb_rmse:.2f} (±{results_df['cb_rmse'].std():.2f})")
    logger.info(f"  MAE:  {avg_cb_mae:.2f} (±{results_df['cb_mae'].std():.2f})")
    logger.info(f"  WAPE: {avg_cb_wape:.2f}% (±{results_df['cb_wape'].std():.2f}%)")
    
    # Save results
    results_df.to_csv(artifacts_dir / "rolling_window_results.csv", index=False)
    logger.info(f"\nSaved: {artifacts_dir / 'rolling_window_results.csv'}")
    
    return {
        "windows": windows_results,
        "avg_xgb_rmse": avg_xgb_rmse,
        "avg_xgb_mae": avg_xgb_mae,
        "avg_xgb_wape": avg_xgb_wape,
        "avg_cb_rmse": avg_cb_rmse,
        "avg_cb_mae": avg_cb_mae,
        "avg_cb_wape": avg_cb_wape,
    }

