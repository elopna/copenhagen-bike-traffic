# Data Download Instructions

This directory should contain the following files:

## Required Files

1. **Bike Traffic Data** (from Kaggle):
   - `rides.csv` (77 MB)
   - `total_rides.csv` (22 MB)
   - `road_info.csv` (4 KB)
   
   Download from: https://www.kaggle.com/datasets/emilhvitfeldt/bike-traffic-counts-in-copenhagen

2. **Weather Data** (from Open-Meteo):
   - `open-meteo-55.71N12.44E6m.csv` (59 MB)
   
   Download from: https://open-meteo.com/en/docs/historical-weather-api
   
   Or use this direct link with parameters:
   ```
   https://archive-api.open-meteo.com/v1/archive?latitude=55.68647259287289,55.65052205120987,55.63834706611601,55.69228869871818,55.70176099776455,55.72019722193979,55.69366342867243,55.667689429115626,55.672726099036026,55.67024734093181,55.71834847510562,55.661664802305424,55.68337519542479&longitude=12.564408128593891,12.510872019529161,12.599838269497782,12.568790322179641,12.533557598251525,12.49482539698566,12.548061150106621,12.570875705677551,12.478559242110604,12.595037788980418,12.53986751879737,12.518010357423124,12.555881921044719&start_date=2005-01-01&end_date=2014-12-31&hourly=temperature_2m,wind_speed_10m,rain,snowfall&format=csv
   ```

## File Structure

```
data/
├── README.md (this file)
├── rides.csv
├── total_rides.csv
├── road_info.csv
└── open-meteo-55.71N12.44E6m.csv
```
