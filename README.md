# 🔋 Predicting Energy Consumption with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-yellow.svg)](https://lightgbm.readthedocs.io/)

A comprehensive machine learning project for predicting household energy consumption in London using weather data and time-series features. This project implements advanced gradient boosting models (XGBoost and LightGBM) with lag features for accurate energy consumption forecasting.

## 📊 Project Overview

This project predicts daily energy consumption using:
- **Historical energy consumption data** (lag features)
- **Weather data** (temperature, precipitation, cloud cover, etc.)
- **Temporal features** (day of week, month, quarter, etc.)

### Key Features
- ✅ Time-series analysis with autocorrelation and partial autocorrelation plots
- ✅ Lag feature engineering (9 optimized lag features)
- ✅ Two state-of-the-art models: XGBoost and LightGBM
- ✅ Comprehensive hyperparameter tuning with GridSearchCV
- ✅ Time-series cross-validation (TimeSeriesSplit)
- ✅ Multiple evaluation metrics (MAE, MSE, RMSE, MAPE)
- ✅ Feature importance analysis

## 📁 Project Structure

```
energy_consumption/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore file
│
├── London_energy.ipynb                # Initial EDA and data exploration
├── adding_lagFeatures.ipynb           # Lag feature analysis (original)
├── rolling_satisticMoving_avg.ipynb   # Rolling statistics analysis
├── energy_prediction_optimized.py     # ⭐ Optimized production script
│
└── Output Files (generated):
    ├── predictions_with_lags.csv      # Model predictions
    ├── model_comparison_with_lags.csv # Performance comparison
    └── feature_importance_xgb.csv     # Feature importance rankings
```

## 📈 Dataset

### Data Sources
1. **[London Homes Energy Data](https://www.kaggle.com/datasets/emmanuelfwerr/london-homes-energy-data)** - Historical energy consumption
2. **[London Weather Data](https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data)** - Weather features

### Features Used
- **Temporal Features**: `date`, `day_of_week`, `day_of_year`, `month`, `quarter`, `year`
- **Weather Features**: `cloud_cover`, `sunshine`, `global_radiation`, `max_temp`, `mean_temp`, `min_temp`, `precipitation`, `pressure`, `snow_depth`
- **Lag Features**: `lag_1` through `lag_9` (previous 9 days of consumption)

### Data Split
- **Training Period**: November 2011 - June 2013 (586 samples)
- **Testing Period**: July 2013 - March 2014 (243 samples)

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/rahulsingh1397/Predicting_Energy_Consumption.git
cd Predicting_Energy_Consumption/energy_consumption
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
- Download from Kaggle: [London Energy & Weather Data](https://www.kaggle.com/datasets/emmanuelfwerr/london-homes-energy-data)
- Place `df_avg_consumption.csv` in: `E:\Projects\Python\Predicting_Energy_Consumption_dataset\`
- Or update the path in the script to your local path

### Usage

#### Option 1: Run the Optimized Script (Recommended)
```bash
python energy_prediction_optimized.py
```

This will:
- Load and preprocess the data
- Create 9 lag features
- Train both XGBoost and LightGBM models with optimized hyperparameters
- Perform time-series cross-validation
- Output comprehensive evaluation metrics
- Save predictions and results to CSV files

#### Option 2: Run Jupyter Notebooks
```bash
jupyter notebook
```
Then open:
- `London_energy.ipynb` - For initial data exploration
- `adding_lagFeatures.ipynb` - For lag feature analysis
- `rolling_satisticMoving_avg.ipynb` - For rolling statistics

## 🎯 Model Performance

### Performance Metrics (with 9 Lag Features)

| Model    | MAE    | MSE    | RMSE   | MAPE   |
|----------|--------|--------|--------|--------|
| XGBoost  | 0.3845 | 0.7600 | 0.8718 | 0.1478 |
| LightGBM | 0.4119 | 0.7712 | 0.8782 | 0.1498 |

### Improvements Over Baseline
- **XGBoost**: ~45% reduction in MAE compared to baseline without lag features
- **LightGBM**: ~41% reduction in MAE compared to baseline without lag features

### Best Hyperparameters

**XGBoost:**
```python
{
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'max_depth': 3,
    'n_estimators': 300,
    'subsample': 0.9,
    'min_child_weight': 3
}
```

**LightGBM:**
```python
{
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 300,
    'num_leaves': 50,
    'min_child_samples': 20
}
```

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Analyzed energy consumption patterns over time
- Identified seasonal trends and weather correlations
- Visualized autocorrelation and partial autocorrelation

### 2. Feature Engineering
- **Lag Features**: Created 9 lag features based on PACF analysis
- **Temporal Features**: Extracted day, month, quarter, year
- **Weather Features**: Normalized and cleaned weather data

### 3. Model Selection
- Chose gradient boosting models for their:
  - Ability to handle non-linear relationships
  - Built-in feature importance
  - Robustness to missing values
  - Strong performance on time-series data

### 4. Hyperparameter Optimization
- Used `GridSearchCV` with `TimeSeriesSplit` (5 folds)
- Optimized for negative mean squared error
- Tested 162+ parameter combinations per model

### 5. Model Evaluation
- Multiple metrics: MAE, MSE, RMSE, MAPE
- Time-series cross-validation to prevent data leakage
- Feature importance analysis

## 📊 Key Insights

### Top 5 Most Important Features (XGBoost)
1. `lag_1` - Previous day's consumption (highest importance)
2. `lag_2` - 2 days ago consumption
3. `mean_temp` - Average temperature
4. `lag_3` - 3 days ago consumption
5. `max_temp` - Maximum temperature

### Findings
- **Lag features are crucial**: The most recent consumption values are the strongest predictors
- **Temperature matters**: Both mean and max temperature significantly impact consumption
- **Seasonal patterns**: Quarter and month features show moderate importance
- **Weather conditions**: Cloud cover and precipitation have minor but measurable effects

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Time Series**: statsmodels
- **Development**: Jupyter Notebook

## 📝 Requirements

See `requirements.txt` for full dependencies:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
statsmodels
jupyter
```

## 🔄 Future Improvements

- [ ] Add LSTM/GRU neural networks for comparison
- [ ] Implement ensemble methods (stacking/blending)
- [ ] Add more weather features (wind speed, humidity)
- [ ] Extend prediction horizon (multi-step forecasting)
- [ ] Deploy model as REST API
- [ ] Add real-time prediction capabilities
- [ ] Implement automated retraining pipeline

## 🐛 Bug Fixes in This Version

### Fixed Issues:
1. ✅ **Variable name bug** in cell 5: Changed `df["consumption"]` to `df_avg_consumption["consumption"]`
2. ✅ **Missing lag features**: Added all 9 lag features to training/testing datasets
3. ✅ **Incomplete feature set**: Ensured X_train and X_test include lag features
4. ✅ **Suboptimal hyperparameters**: Expanded grid search with additional parameters

## 📄 License

This project is for educational purposes. Please refer to the dataset licenses on Kaggle for usage terms.

## 👤 Author

**Rahul Singh**
- GitHub: [@rahulsingh1397](https://github.com/rahulsingh1397)

## 🙏 Acknowledgments

- Dataset providers on Kaggle
- XGBoost and LightGBM development teams
- Open-source community

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ **If you find this project helpful, please consider giving it a star!** ⭐
