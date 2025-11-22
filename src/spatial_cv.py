"""Spatial cross-validation using H3 hexagonal blocks."""

import logging
from pathlib import Path
from typing import List, Tuple
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold


logger = logging.getLogger(__name__)


def make_spatial_folds(
    df: pd.DataFrame, n_splits: int = 10, h3_resolution: int = 7
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create spatial cross-validation folds using H3 blocks.
    
    Each fold validates on entire spatial blocks (H3 cells at resolution 7)
    that were not seen during training.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with h3_r7 column (H3 index at resolution 7)
    n_splits : int
        Number of folds
    h3_resolution : int
        H3 resolution for blocks (default 7 = ~5.16 km² hexagons)
        
    Returns
    -------
    folds : list of tuples
        List of (train_indices, valid_indices) for each fold
    """
    h3_col = f"h3_r{h3_resolution}"
    
    if h3_col not in df.columns:
        raise ValueError(
            f"Column '{h3_col}' not found. "
            f"Please ensure features.py has added H3 at resolution {h3_resolution}."
        )
    
    logger.info(f"Creating {n_splits} spatial folds using {h3_col}...")
    
    # Get H3 blocks
    blocks = df[h3_col].values
    unique_blocks = np.unique(blocks)
    
    logger.info(f"Number of unique H3 blocks: {len(unique_blocks)}")
    
    if len(unique_blocks) < n_splits:
        logger.warning(
            f"Only {len(unique_blocks)} unique blocks, but {n_splits} folds requested. "
            f"Some folds will be small."
        )
    
    # Use GroupKFold to ensure entire blocks go to train or validation
    gkf = GroupKFold(n_splits=n_splits)
    
    folds = []
    for fold_idx, (train_idx, valid_idx) in enumerate(gkf.split(df, groups=blocks)):
        train_blocks = set(blocks[train_idx])
        valid_blocks = set(blocks[valid_idx])
        
        logger.info(
            f"Fold {fold_idx + 1}/{n_splits}: "
            f"train {len(train_blocks)} blocks ({len(train_idx):,} rows), "
            f"valid {len(valid_blocks)} blocks ({len(valid_idx):,} rows)"
        )
        
        folds.append((train_idx, valid_idx))
    
    return folds


def save_folds(folds: List[Tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    """
    Save fold indices to pickle file.
    
    Parameters
    ----------
    folds : list of tuples
        List of (train_indices, valid_indices)
    output_path : Path
        Output pickle file path
    """
    with open(output_path, "wb") as f:
        pickle.dump(folds, f)
    logger.info(f"Saved folds to {output_path}")


def load_folds(folds_path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load fold indices from pickle file.
    
    Parameters
    ----------
    folds_path : Path
        Path to folds pickle file
        
    Returns
    -------
    folds : list of tuples
        List of (train_indices, valid_indices)
    """
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)
    logger.info(f"Loaded {len(folds)} folds from {folds_path}")
    return folds


def spatial_cv_pipeline(artifacts_dir: Path, cv_folds: int = 10) -> None:
    """
    Create and save spatial CV folds.
    
    Parameters
    ----------
    artifacts_dir : Path
        Directory with features.parquet
    cv_folds : int
        Number of spatial folds
    """
    logger.info("Loading features data...")
    df = pd.read_parquet(artifacts_dir / "features.parquet")
    
    logger.info(f"Data shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Create folds
    folds = make_spatial_folds(df, n_splits=cv_folds, h3_resolution=7)
    
    # Save folds
    save_folds(folds, artifacts_dir / "folds.pkl")
    
    logger.info("\nSpatial CV folds created successfully!")

