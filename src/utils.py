"""Utility functions for logging, reproducibility, and metrics."""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging with timestamp and level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    # Note: XGBoost and CatBoost seeds set in model parameters


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10, min_value: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error (in %).
    
    Only computed for observations where y_true >= min_value to avoid division by zero.
    """
    mask = y_true >= min_value
    if mask.sum() == 0:
        return np.nan
    return float(100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).
    
    WAPE = sum(|y_true - y_pred|) / sum(y_true) * 100
    
    Better than MAPE for data with zeros or high variance because:
    - Weighted by actual volume (gives more weight to high-traffic stations)
    - No division by individual observations (avoids inf for zeros)
    - Symmetric treatment of over/under predictions
    
    Parameters
    ----------
    y_true : array
        True values
    y_pred : array
        Predicted values
        
    Returns
    -------
    float
        WAPE (in %)
    """
    return float(100 * np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute all metrics: RMSE, MAE, MAPE, WAPE.
    
    Parameters
    ----------
    y_true : array
        True values
    y_pred : array
        Predicted values
        
    Returns
    -------
    dict
        Dictionary with metrics: rmse, mae, mape, wape
    """
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred, min_value=10.0),  # Only for y_true >= 10
        "wape": wape(y_true, y_pred),  # Weighted by volume
    }
    
    return metrics


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """Pretty print metrics dictionary."""
    logger = logging.getLogger(__name__)
    if prefix:
        logger.info(f"{prefix}:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key.upper()}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

