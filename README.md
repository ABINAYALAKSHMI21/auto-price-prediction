# 🚗 Auto Price Prediction

A regression project that models the price of automobiles from 25+ technical and categorical specifications (make, engine specs, body style, mileage, etc.), so that management can understand how design and engineering choices drive price — and target specific price points in a new market.

## Problem Statement

An automotive company entering the US market needs to understand what factors make a car expensive there, since pricing dynamics differ across markets. This project builds a predictive model that:
- Identifies which variables significantly affect car price
- Quantifies how well those variables describe price
- Gives management a data-driven way to reverse-engineer design/pricing decisions to hit target price levels

## Dataset

- **Source:** [1985 Auto Imports Database](https://archive.ics.uci.edu/dataset/10/automobile) (UCI Machine Learning Repository)
- **Size:** 205 rows × 26 columns
- **Target variable:** `price` (continuous, $5,118–$45,400)
- **Features:** symboling, normalized-losses, make, fuel-type, aspiration, num-of-doors, body-style, drive-wheels, engine-location, wheel-base, length, width, height, curb-weight, engine-type, num-of-cylinders, engine-size, fuel-system, bore, stroke, compression-ratio, horsepower, peak-rpm, city-mpg, highway-mpg

## Project Workflow

| Stage | What was done |
|---|---|
| **1. Data Cleaning** | Handled `?` placeholders (corrupted values) in normalized-losses, bore, stroke, horsepower, peak-rpm, and num-of-doors via median/mode imputation; corrected data types |
| **2. EDA** | Automated profiling report, univariate/bivariate/multivariate analysis on numerical and categorical features |
| **3. Preprocessing** | Label encoding for binary categoricals (fuel-type, aspiration, num-of-doors, engine-location); one-hot encoding for multi-class categoricals (body-style, drive-wheels, engine-type, num-of-cylinders, fuel-system, make); Min-Max scaling for numerical features; correlation analysis |
| **4. Dimensionality Reduction** | Principal Component Analysis (PCA) |
| **5. Modeling** | Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, K-Nearest Neighbors, and an Artificial Neural Network (MLPRegressor) — each with hyperparameter tuning (GridSearchCV / RandomizedSearchCV) |
| **6. Evaluation** | Compared all models on R² score |

## Results

All seven models were compared on R² score after tuning. **Decision Tree Regressor** delivered the strongest predictive performance on this dataset, with the MLP Regressor (ANN) close behind. Full metrics and the comparison plot are in the notebook (Section 10).

## Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `XGBoost` · `ydata-profiling`

## Repository Structure

```
auto-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── Auto_Price_Prediction.ipynb
├── data/
│   └── auto_imports.csv
└── reports/
    └── Model_Comparison_and_Challenges.md
```

## How to Run

```bash
git clone https://github.com/<your-username>/auto-price-prediction.git
cd auto-price-prediction
pip install -r requirements.txt
jupyter notebook notebook/Auto_Price_Prediction.ipynb
```

## Key Challenges

- Extracting meaningful insight from a dataset with mixed categorical/continuous features and hidden corrupted values (`?`)
- Cleaning and correctly re-typing corrupted numeric columns
- Long training/tuning time for ensemble and boosting models
- Finding a `random_state` at which all models could be fairly compared

## Author

*Add your name and links here (LinkedIn / portfolio) so recruiters can find you.*
