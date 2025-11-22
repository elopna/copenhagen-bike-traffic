# Quick Setup Guide

## 1. Clone Repository
```bash
git clone ...
cd cph-aadbt-mini
```

## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Download Data
See `data/README.md` for detailed instructions.

**Quick download:**
1. Kaggle data: https://www.kaggle.com/datasets/emilhvitfeldt/bike-traffic-counts-in-copenhagen
   - Download `rides.csv`, `total_rides.csv`, `road_info.csv`
   - Place in `data/` directory

2. Weather data: Use Open-Meteo API (see `data/README.md` for direct link)
   - Save as `open-meteo-55.71N12.44E6m.csv` in `data/` directory

## 4. Run Pipeline

### Option A: Jupyter Notebook (Recommended for Demo)
```bash
cd notebooks
jupyter notebook 00_demo.ipynb
```

### Option B: Command Line
```bash
# Full pipeline
python run.py all --freq hour --test_size 0.2 --neighbors 4 --cv_folds 5

# Or step by step:
python run.py prepare --freq hour --test_size 0.2
python run.py fe --neighbors 4
python run.py train
python run.py explain
python run.py map
```

## 5. View Results

Outputs will be saved to `artifacts/`:
- `cph_aadbt_map.html` - Interactive map
- `shap_summary.png` - Feature importance
- `test_predictions.parquet` - Hourly predictions
- `cv_metrics.csv` - Cross-validation results

