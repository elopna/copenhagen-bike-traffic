# Copenhagen Bike Ridership Forecasting

> Spatial-temporal forecasting pipeline for hourly bike traffic prediction at Copenhagen counting stations (2005-2014)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project implements a complete machine learning pipeline for predicting hourly bicycle traffic at 13 counting stations in Copenhagen. The model achieves **WAPE 19-23%** using spatial-temporal features, weather data, and ensemble methods.

**Key Features:**
- 🗺️ **H3 spatial indexing** for geographic feature encoding
- 🌤️ **Weather integration** (temperature, wind, precipitation)
- 📅 **Danish holidays** and temporal cyclicity features
- ⏱️ **Lagged features** (24h lag + 7-day rolling mean)
- 🔒 **Data leakage prevention** in cross-validation
- 🎯 **Spatial CV** (H3-based) + Rolling window validation
- 🤖 **Ensemble models** (XGBoost + CatBoost)
- 📊 **SHAP explainability** + Interactive maps

## 📊 Results

| Model | WAPE | MAE (bikes/hour) | Validation |
|-------|------|------------------|------------|
| XGBoost | 19.22% ± 5.66% | 50.87 ± 16.32 | Rolling window (8 folds) |
| CatBoost | 23.09% ± 6.77% | 61.21 ± 20.61 | Rolling window (8 folds) |
| Final Test | 23.16% | 60.11 | Hold-out 20% (2012-2014) |

**Validation Strategy:** Rolling window with 2-year training and 1-year test periods across 2005-2014.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/elopna/cph-bike-ridership-forecasting.git
cd cph-bike-ridership-forecasting
pip install -r requirements.txt
```

### Data Download
See [`data/README.md`](data/README.md) for instructions:
1. **Kaggle data**: [Copenhagen bike counters](https://www.kaggle.com/datasets/emilhvitfeldt/bike-traffic-counts-in-copenhagen)
2. **Weather data**: [Open-Meteo API](https://open-meteo.com/en/docs/historical-weather-api)

### Run Pipeline
```bash
# Option 1: Full pipeline
python run.py all --freq hour --test_size 0.2 --neighbors 4 --cv_folds 5

# Option 2: Step by step
python run.py prepare --freq hour --test_size 0.2
python run.py fe --neighbors 4
python run.py train
python run.py explain
python run.py map
```

### Or use Jupyter Notebook
```bash
jupyter notebook notebooks/00_demo.ipynb
```

## 📁 Project Structure

```
.
├── src/                          # Source modules
│   ├── prepare_data.py          # Data loading & schema detection
│   ├── features.py              # H3, spatial, temporal features
│   ├── advanced_features.py     # Weather, holidays, lags
│   ├── spatial_cv.py            # H3-based cross-validation
│   ├── models.py                # XGBoost + CatBoost ensemble
│   ├── explain.py               # SHAP analysis
│   └── map_viz.py               # Interactive Folium maps
├── notebooks/
│   └── 00_demo.ipynb            # Full pipeline demo with results
├── data/                         # Data files (download separately)
├── artifacts/                    # Model outputs (generated)
├── run.py                        # CLI interface
└── requirements.txt              # Dependencies
```

## 🧪 Methodology

### Features (38 total)
- **Spatial:** H3 hexagon indices (r7, r8), distance to city center, KNN neighbor statistics
- **Temporal:** Hour/day cyclicity (sin/cos), day of week, month, is_weekend
- **Weather:** Temperature, wind speed, precipitation, snowfall
- **Contextual:** Danish public holidays
- **Lagged:** 24-hour lag, 7-day rolling mean

### Models
1. **XGBoost:** Baseline with one-hot encoded H3
2. **CatBoost:** Ensemble with native H3 encoding + neighbor OOF lags

### Validation
- **Spatial CV:** 5 folds using H3 blocks (prevents geographic leakage)
- **Rolling Window:** 2-year train → 1-year test (8 windows, 2005-2014)
- **Data Leakage Prevention:** Lags computed per-fold from train+validation concatenation

## 🔍 Key Insights

From SHAP analysis (top features):
1. `count_rolling_mean_7d` (107.74) - Historical pattern
2. `count_lag_24h` (64.29) - Yesterday same hour
3. `hour_cos` (55.33) - Time of day
4. `h3_r8` (21.62) - Location
5. `dow` (21.37) - Day of week

**Peak hours:** 8:00 AM and 4:00 PM (commuting patterns correctly captured)

## ⚠️ Practical Considerations

**Prediction Horizon:**
- Model uses 24h lag → suitable for **day-ahead forecasting**
- Weather forecasts (not historical data) needed in production
- Longer horizons (7+ days) require recursive prediction or alternative approaches

**Limitations:**
- Weather forecast quality impacts accuracy
- Special events not captured (except standard holidays)
- Single city model (transfer learning to other cities untested)

## 🔮 Future Improvements

- [ ] Event calendar integration (concerts, sports, construction)
- [ ] Station-specific weather interpolation
- [ ] Deep learning models (LSTM/Transformer)
- [ ] Real-time traffic/route availability data
- [ ] Transfer learning to other cities
- [ ] Ensemble with weather forecast uncertainty

## 📚 Data Sources

- **Bike Traffic:** [Kaggle - Copenhagen Bike Counters](https://www.kaggle.com/datasets/emilhvitfeldt/bike-traffic-counts-in-copenhagen)
- **Weather:** [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api)
- **Holidays:** Python `holidays` library (Denmark)

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 👤 Author

Eduard Lopatin - [GitHub](https://github.com/elopna)

---

**⭐ If this project helps your research, please consider citing or starring!**
