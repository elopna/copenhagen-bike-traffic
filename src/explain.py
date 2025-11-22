"""Model explainability using SHAP."""

import logging
import pickle
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


def compute_shap_values(
    model,
    X_sample: pd.DataFrame,
    max_samples: int = 50000,
) -> Tuple[shap.Explanation, pd.DataFrame]:
    """
    Compute SHAP values for model.
    
    Parameters
    ----------
    model : CatBoostRegressor
        Trained model
    X_sample : DataFrame
        Sample data for SHAP computation
    max_samples : int
        Maximum number of samples to use
        
    Returns
    -------
    shap_values : shap.Explanation
        SHAP values
    importance_df : DataFrame
        Feature importance dataframe
    """
    logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
    
    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    logger.info("SHAP computation complete")
    
    # Compute feature importance (mean absolute SHAP)
    importance = np.abs(shap_values.values).mean(axis=0)
    
    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)
    
    return shap_values, importance_df


def plot_shap_summary(
    shap_values: shap.Explanation,
    output_path: Path,
    max_display: int = 20,
) -> None:
    """
    Plot SHAP summary plot.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values
    output_path : Path
        Output path for plot
    max_display : int
        Maximum number of features to display
    """
    logger.info("Creating SHAP summary plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved: {output_path}")


def explain_pipeline(
    artifacts_dir: Path,
    max_samples: int = 50000,
) -> None:
    """
    Full explainability pipeline.
    
    Parameters
    ----------
    artifacts_dir : Path
        Directory with model_results.pkl and features.parquet
    max_samples : int
        Maximum samples for SHAP computation
    """
    logger.info("Loading model and data...")
    
    with open(artifacts_dir / "model_results.pkl", "rb") as f:
        results = pickle.load(f)
    
    final_model = results["final_model"]
    cb_oof_preds = results["cb_oof_preds"]
    
    train_df = pd.read_parquet(artifacts_dir / "features.parquet")
    
    # Use CatBoost OOF predictions as neighbor lag (already includes spatial info)
    logger.info("Using CatBoost OOF predictions as spatial_oof_neighbors_mean for SHAP...")
    
    # Load neighbor graph for simple averaging
    with open(artifacts_dir / "neighbor_graph.pkl", "rb") as f:
        neighbor_graph = pickle.load(f)
    neighbor_indices = neighbor_graph["indices"]
    
    # Vectorized neighbor lag creation (simplified)
    logger.info("Creating simplified neighbor lags for SHAP...")
    train_df["_oof_pred"] = cb_oof_preds
    
    # For each site, compute mean OOF of its neighbors (ignore time for speed)
    site_neighbor_means = {}
    for site_id, neighbors in neighbor_indices.items():
        if len(neighbors) > 0:
            neighbor_oof = train_df[train_df["site_id"].isin(neighbors)]["_oof_pred"].mean()
            site_neighbor_means[site_id] = neighbor_oof
        else:
            site_neighbor_means[site_id] = train_df["_oof_pred"].median()
    
    train_df["spatial_oof_neighbors_mean"] = train_df["site_id"].map(site_neighbor_means)
    train_df["spatial_oof_neighbors_mean"] = train_df["spatial_oof_neighbors_mean"].fillna(train_df["_oof_pred"].median())
    train_df = train_df.drop(columns=["_oof_pred"])
    
    # Add lagged features if not present
    if "count_lag_24h" not in train_df.columns:
        logger.info("Adding lagged features for SHAP...")
        from src.advanced_features import add_lagged_features
        train_df = add_lagged_features(train_df, "count")
    
    # Sample data for SHAP
    if len(train_df) > max_samples:
        logger.info(f"Sampling {max_samples} rows for SHAP (from {len(train_df):,} total)...")
        
        # Stratified sampling by H3 block and hour
        sample_df = train_df.groupby(["h3_r7", "hour"], group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, max_samples // 100)), random_state=42)
        )
        
        if len(sample_df) > max_samples:
            sample_df = sample_df.sample(max_samples, random_state=42)
    else:
        sample_df = train_df
    
    logger.info(f"Using {len(sample_df):,} samples for SHAP analysis")
    
    # Prepare features (use CatBoost features which include h3_r8)
    from src.models import FEATURE_COLS_CATBOOST_WITH_LAG

    # Filter features to only those present in data
    available_features = [f for f in FEATURE_COLS_CATBOOST_WITH_LAG if f in sample_df.columns]
    missing_features = [f for f in FEATURE_COLS_CATBOOST_WITH_LAG if f not in sample_df.columns]
    
    if missing_features:
        logger.warning(f"Missing features (will be excluded from SHAP): {missing_features}")
    
    X_sample = sample_df[available_features].copy()
    
    # Compute SHAP values
    shap_values, importance_df = compute_shap_values(final_model, X_sample, max_samples)
    
    # Save feature importance
    importance_df.to_csv(artifacts_dir / "shap_top_features.csv", index=False)
    logger.info(f"Saved: {artifacts_dir / 'shap_top_features.csv'}")
    
    # Print top features
    logger.info("\nTop 10 Features by SHAP Importance:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    # Plot SHAP summary
    plot_shap_summary(shap_values, artifacts_dir / "shap_summary.png", max_display=20)
    
    logger.info("\nExplainability analysis complete!")

