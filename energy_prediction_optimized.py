"""
Optimized Energy Consumption Prediction with Lag Features
Fixes:
1. Fixed df vs df_avg_consumption bug
2. Added lag features to training/testing data
3. Optimized hyperparameters
4. Added comprehensive evaluation metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.offsetbox import AnchoredText
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns
import math
import lightgbm as lgb

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Loading data...")
df_avg_consumption = pd.read_csv(r"E:\Projects\Python\Predicting_Energy_Consumption_dataset\df_avg_consumption.csv")
print(f"Data shape: {df_avg_consumption.shape}")
print(f"Columns: {df_avg_consumption.columns.tolist()}")

# ============================================================================
# 2. VISUALIZATION FUNCTIONS
# ============================================================================
def plot_lag(x, lag=1, ax=None, **kwargs):
    """Plot lag correlation"""
    x_ = x.shift(lag)
    y_ = x
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(alpha=0.75, s=3)
    line_kws = dict(color='C3')

    ax = sns.regplot(x=x_, y=y_, scatter_kws=scatter_kws, 
                     line_kws=line_kws, lowess=True, ax=ax, **kwargs)
    
    # Adding correlation on plot
    at = AnchoredText(
        f"{y_.corr(x_):.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax

def plot_autocorrelation(x, lags=6, lagplot_kwargs={}, **kwargs):
    """Plot autocorrelation for multiple lags"""
    kwargs.setdefault("nrows", 2)
    kwargs.setdefault("ncols", math.ceil(lags / 2))
    kwargs.setdefault("figsize", (kwargs["ncols"] * 2, 2 * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(2 * kwargs["ncols"])):
        if k + 1 <= lags:
            ax = plot_lag(x, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag #{k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis("off")
    plt.setp(axs[-1, :], xlabel=x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

# ============================================================================
# 3. ANALYZE AUTOCORRELATION (Optional - for visualization)
# ============================================================================
print("\nAnalyzing autocorrelation...")
# Uncomment to visualize:
# _ = plot_autocorrelation(df_avg_consumption["consumption"], lags=12)
# plt.savefig('autocorrelation_plot.png')
# plt.close()

# FIXED: Changed df to df_avg_consumption
# _ = plot_pacf(df_avg_consumption["consumption"], lags=12)
# plt.savefig('pacf_plot.png')
# plt.close()

# ============================================================================
# 4. CREATE LAG FEATURES
# ============================================================================
def create_lag_features(df, lags=9):
    """Create lag features for time series prediction"""
    df_copy = df.copy()
    y = df_copy["consumption"]
    for lag in range(1, lags + 1):
        df_copy[f"lag_{lag}"] = y.shift(lag)
    return df_copy

print("\nCreating lag features...")
# Using 9 lags based on PACF analysis (best choice according to comments)
df_avg_consumption = create_lag_features(df_avg_consumption, lags=9)
print(f"Data shape after adding lags: {df_avg_consumption.shape}")
print(f"New columns: {[col for col in df_avg_consumption.columns if 'lag' in col]}")

# ============================================================================
# 5. TRAIN/TEST SPLIT
# ============================================================================
print("\nSplitting data into train/test sets...")
training_mask = df_avg_consumption["date"] < "2013-07-01"
training_data = df_avg_consumption.loc[training_mask].copy()
print(f"Train shape: {training_data.shape}")

testing_mask = df_avg_consumption["date"] >= "2013-07-01"
testing_data = df_avg_consumption.loc[testing_mask].copy()
print(f"Test shape: {testing_data.shape}")

# Drop date column
training_data = training_data.drop("date", axis=1)
testing_date = testing_data["date"]
testing_data = testing_data.drop("date", axis=1)

# FIXED: Added lag features to X_train and X_test
feature_columns = ["day_of_week", "day_of_year", "month", "quarter", "year",
                   "cloud_cover", "sunshine", "global_radiation", "max_temp",
                   "mean_temp", "min_temp", "precipitation", "pressure",
                   "snow_depth"]

# Add lag features
lag_columns = [f"lag_{i}" for i in range(1, 10)]
feature_columns.extend(lag_columns)

X_train = training_data[feature_columns]
y_train = training_data["consumption"]

X_test = testing_data[feature_columns]
y_test = testing_data["consumption"]

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Features: {feature_columns}")

# ============================================================================
# 6. XGBOOST MODEL WITH OPTIMIZED HYPERPARAMETERS
# ============================================================================
print("\n" + "="*70)
print("TRAINING XGBOOST MODEL")
print("="*70)

cv_split = TimeSeriesSplit(n_splits=5, test_size=100)
XGBmodel = XGBRegressor(random_state=42)

# Optimized parameter grid
param_grid_xgb = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "subsample": [0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5]
}

print("Running GridSearchCV for XGBoost...")
grid_search_xgb = GridSearchCV(
    XGBmodel, 
    param_grid_xgb, 
    cv=cv_split, 
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)
grid_search_xgb.fit(X_train, y_train)

print(f"\nBest XGBoost Parameters: {grid_search_xgb.best_params_}")
print(f"Best CV Score (neg MSE): {grid_search_xgb.best_score_:.4f}")

# Make predictions
y_pred_xgb = grid_search_xgb.predict(X_test)

# Evaluate
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test))

print("\nXGBoost Test Set Performance:")
print(f"MAE:  {mae_xgb:.4f}")
print(f"MSE:  {mse_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.4f}")
print(f"MAPE: {mape_xgb:.4f}")

# ============================================================================
# 7. LIGHTGBM MODEL WITH OPTIMIZED HYPERPARAMETERS
# ============================================================================
print("\n" + "="*70)
print("TRAINING LIGHTGBM MODEL")
print("="*70)

cv_split_lgb = TimeSeriesSplit(n_splits=5, test_size=100)
LightGBM_model = lgb.LGBMRegressor(random_state=42, verbose=-1)

# Optimized parameter grid
param_grid_lgb = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "num_leaves": [31, 50, 70],
    "min_child_samples": [20, 30, 50]
}

print("Running GridSearchCV for LightGBM...")
grid_search_lgb = GridSearchCV(
    LightGBM_model, 
    param_grid_lgb, 
    cv=cv_split_lgb,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)
grid_search_lgb.fit(X_train, y_train)

print(f"\nBest LightGBM Parameters: {grid_search_lgb.best_params_}")
print(f"Best CV Score (neg MSE): {grid_search_lgb.best_score_:.4f}")

# Make predictions
y_pred_lgb = grid_search_lgb.predict(X_test)

# Evaluate
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mse_lgb)
mape_lgb = np.mean(np.abs((y_test - y_pred_lgb) / y_test))

print("\nLightGBM Test Set Performance:")
print(f"MAE:  {mae_lgb:.4f}")
print(f"MSE:  {mse_lgb:.4f}")
print(f"RMSE: {rmse_lgb:.4f}")
print(f"MAPE: {mape_lgb:.4f}")

# ============================================================================
# 8. COMPARISON AND VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

results_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM'],
    'MAE': [mae_xgb, mae_lgb],
    'MSE': [mse_xgb, mse_lgb],
    'RMSE': [rmse_xgb, rmse_lgb],
    'MAPE': [mape_xgb, mape_lgb]
})

print("\n", results_df.to_string(index=False))

# Determine best model
best_model_name = results_df.loc[results_df['MAE'].idxmin(), 'Model']
print(f"\nBest Model (by MAE): {best_model_name}")

# ============================================================================
# 9. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE (XGBoost)")
print("="*70)

feature_importance_xgb = pd.DataFrame({
    'feature': feature_columns,
    'importance': grid_search_xgb.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance_xgb.head(15).to_string(index=False))

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save predictions
predictions_df = pd.DataFrame({
    'date': testing_date.values,
    'actual': y_test.values,
    'xgboost_pred': y_pred_xgb,
    'lightgbm_pred': y_pred_lgb
})
predictions_df.to_csv('predictions_with_lags.csv', index=False)
print("Predictions saved to: predictions_with_lags.csv")

# Save model comparison
results_df.to_csv('model_comparison_with_lags.csv', index=False)
print("Model comparison saved to: model_comparison_with_lags.csv")

# Save feature importance
feature_importance_xgb.to_csv('feature_importance_xgb.csv', index=False)
print("Feature importance saved to: feature_importance_xgb.csv")

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE!")
print("="*70)
print("\nKey Improvements:")
print("1. ✓ Fixed df vs df_avg_consumption bug")
print("2. ✓ Added 9 lag features to training/testing data")
print("3. ✓ Optimized hyperparameters with expanded grid search")
print("4. ✓ Added subsample and min_child_weight to XGBoost")
print("5. ✓ Added min_child_samples to LightGBM")
print("6. ✓ Comprehensive evaluation metrics (MAE, MSE, RMSE, MAPE)")
print("7. ✓ Feature importance analysis")
print("8. ✓ Results saved to CSV files")
