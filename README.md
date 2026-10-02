# Freight Rate Prediction Assessment

This project builds a machine learning model to predict freight load rates using historical shipment data.

## Objective

The goal is to predict `posted_rate` for unseen freight loads using shipment characteristics such as:

- Pickup and delivery locations
- Distance
- Equipment type
- Weight
- Market index
- Quote signal
- Date-related features

## Data Validation Strategy

A time-based validation split was used to better simulate future predictions:

- Training data: January 2025 to September 2025
- Validation data: October 2025

This approach was preferred over a random split because the final task requires predictions for future dates.

## Data Preparation

The following preprocessing steps were applied:

- Converted the `date` column to datetime format
- Extracted:
  - Month
  - Day
  - Day of week
  - Weekend indicator
- Missing `weight` values were filled using the median weight for each equipment type
- Missing `market_index` values were filled using the median market index by month

## Feature Engineering

Additional features were created to improve model performance:

- `is_weekend`
- `weight_per_distance`
- `lat_diff`
- `lon_diff`
- `manhattan_dist`
- `market_quote_ratio`

## Model Experiments

Several regression models were evaluated, including:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost
- LightGBM
- CatBoost Regressor

CatBoost produced the best validation performance.

## Final Model

The final model uses:

- CatBoost Regressor
- Safe feature engineering
- Group-based missing value imputation
- Early stopping during validation

Best time-based validation result:

**MAE: approximately $107.53**

The final model was then retrained using the full labeled development dataset before generating the submission predictions.

## Output Files

The repository includes:

- `Final_code_(freight_rate_model).ipynb` — final training and prediction pipeline
- `freight_rate_ml_Baseline.ipynb` — model experiments and validation
- `validation_predictions.csv` — predictions for the 12,000 validation loads
- `december_predictions.csv` — predictions for the fixed December scenario
- `candidate_december.png` — December prediction chart
- `score.py` — provided output validation script
- `requirements.txt` — Python dependencies

## Validation Output

The provided scoring script successfully validated:

- 12,000 final predictions
- 31 December predictions

It also generated the December prediction chart.

## Run Instructions

Install dependencies:

```bash
pip install -r requirements.txt
