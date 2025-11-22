#!/usr/bin/env python
"""CLI orchestrator for Copenhagen bike traffic modeling pipeline."""

import argparse
import sys
from pathlib import Path

from src.utils import setup_logging, set_seeds
from src.prepare_data import prepare_data_pipeline
from src.features import feature_engineering_pipeline, apply_features_to_test
from src.advanced_features import advanced_features_pipeline
from src.spatial_cv import spatial_cv_pipeline
from src.models import models_pipeline
from src.explain import explain_pipeline
from src.map_viz import map_viz_pipeline


# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Copenhagen Hourly Bike Traffic Modeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline step-by-step
  python run.py prepare --freq hour --test_size 0.2
  python run.py fe --neighbors 4     # Includes advanced features
  python run.py train
  python run.py explain
  python run.py map
  
  # Or run all stages at once
  python run.py all --freq hour --test_size 0.2 --neighbors 4 --cv_folds 5
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage")
    
    # Prepare stage
    prepare_parser = subparsers.add_parser(
        "prepare", help="Data preparation: load, clean, split"
    )
    prepare_parser.add_argument(
        "--freq", default="hour", choices=["hour"], help="Time frequency (default: hour)"
    )
    prepare_parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    
    # Feature engineering stage
    fe_parser = subparsers.add_parser("fe", help="Feature engineering")
    fe_parser.add_argument(
        "--neighbors", type=int, default=10, help="Number of neighbors for KNN (default: 10)"
    )
    
    # Training stage
    train_parser = subparsers.add_parser("train", help="Model training")
    # Note: cv_folds and neighbors are loaded from saved artifacts (folds.pkl, neighbor_graph.pkl)
    
    # Explain stage
    explain_parser = subparsers.add_parser("explain", help="Model explainability with SHAP")
    explain_parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help="Max samples for SHAP (default: 50000)",
    )
    
    # Map stage
    map_parser = subparsers.add_parser("map", help="Create AADBT map visualization")
    
    # All stages
    all_parser = subparsers.add_parser("all", help="Run all stages sequentially")
    all_parser.add_argument(
        "--freq", default="hour", choices=["hour"], help="Time frequency (default: hour)"
    )
    all_parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    all_parser.add_argument(
        "--neighbors", type=int, default=10, help="Number of neighbors (default: 10)"
    )
    all_parser.add_argument(
        "--cv_folds", type=int, default=10, help="Number of spatial CV folds (default: 10)"
    )
    all_parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help="Max samples for SHAP (default: 50000)",
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup
    logger = setup_logging()
    set_seeds(42)
    
    logger.info("=" * 80)
    logger.info("Copenhagen Hourly Bike Traffic Modeling Pipeline")
    logger.info("=" * 80)
    logger.info(f"Command: {args.command}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Artifacts directory: {ARTIFACTS_DIR}")
    logger.info("")
    
    # Execute command
    try:
        if args.command == "prepare":
            logger.info("STAGE: Data Preparation")
            logger.info(f"  Frequency: {args.freq}")
            logger.info(f"  Test size: {args.test_size}")
            logger.info("")
            
            prepare_data_pipeline(DATA_DIR, ARTIFACTS_DIR, args.freq, args.test_size)
        
        elif args.command == "fe":
            logger.info("STAGE: Feature Engineering")
            logger.info(f"  Neighbors: {args.neighbors}")
            logger.info("")
            
            # Train features
            feature_engineering_pipeline(ARTIFACTS_DIR, args.neighbors)
            
            # Apply to test
            logger.info("\nApplying features to test set...")
            apply_features_to_test(ARTIFACTS_DIR)
            
            # Advanced features (weather, holidays, etc.)
            logger.info("\nAdding advanced features...")
            weather_path = DATA_DIR / "open-meteo-55.71N12.44E6m.csv"
            advanced_features_pipeline(
                ARTIFACTS_DIR,
                weather_path,
                location_id=0,
                add_lags=False  # Lags added during training to avoid leakage
            )
            
            # Create CV folds
            logger.info("\nCreating spatial CV folds...")
            spatial_cv_pipeline(ARTIFACTS_DIR, cv_folds=10)
        
        elif args.command == "train":
            logger.info("STAGE: Model Training")
            logger.info("")
            
            models_pipeline(ARTIFACTS_DIR)
        
        elif args.command == "explain":
            logger.info("STAGE: Model Explainability")
            logger.info(f"  Max samples: {args.max_samples}")
            logger.info("")
            
            explain_pipeline(ARTIFACTS_DIR, args.max_samples)
        
        elif args.command == "map":
            logger.info("STAGE: Map Visualization")
            logger.info("")
            
            map_viz_pipeline(ARTIFACTS_DIR)
        
        elif args.command == "all":
            logger.info("STAGE: Running all stages sequentially")
            logger.info(f"  Frequency: {args.freq}")
            logger.info(f"  Test size: {args.test_size}")
            logger.info(f"  Neighbors: {args.neighbors}")
            logger.info(f"  CV folds: {args.cv_folds}")
            logger.info("")
            
            # 1. Prepare
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 1/5: Data Preparation")
            logger.info("=" * 80)
            prepare_data_pipeline(DATA_DIR, ARTIFACTS_DIR, args.freq, args.test_size)
            
            # 2. Feature engineering
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 2/6: Feature Engineering")
            logger.info("=" * 80)
            feature_engineering_pipeline(ARTIFACTS_DIR, args.neighbors)
            apply_features_to_test(ARTIFACTS_DIR)
            
            # 3. Advanced features
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 3/6: Advanced Features")
            logger.info("=" * 80)
            weather_path = DATA_DIR / "open-meteo-55.71N12.44E6m.csv"
            advanced_features_pipeline(
                ARTIFACTS_DIR,
                weather_path,
                location_id=0,
                add_lags=False
            )
            
            # 4. Spatial CV
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 4/6: Spatial Cross-Validation")
            logger.info("=" * 80)
            spatial_cv_pipeline(ARTIFACTS_DIR, args.cv_folds)
            
            # 5. Training
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 5/6: Model Training")
            logger.info("=" * 80)
            models_pipeline(ARTIFACTS_DIR)
            
            # 6. Explain
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 6/6: Model Explainability & Visualization")
            logger.info("=" * 80)
            explain_pipeline(ARTIFACTS_DIR, args.max_samples)
            map_viz_pipeline(ARTIFACTS_DIR)
            
            logger.info("\n" + "=" * 80)
            logger.info("ALL STAGES COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"\nOutputs saved to: {ARTIFACTS_DIR}")
            logger.info(f"  - Map: {ARTIFACTS_DIR / 'cph_aadbt_map.html'}")
            logger.info(f"  - SHAP: {ARTIFACTS_DIR / 'shap_summary.png'}")
            logger.info(f"  - Predictions: {ARTIFACTS_DIR / 'test_predictions.parquet'}")
        
        logger.info("\n✓ Pipeline stage completed successfully!")
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

