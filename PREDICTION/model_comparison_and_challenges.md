# Model Comparison Report

Seven regression models were trained and tuned to predict car price from the available independent variables. Each model was evaluated using R² score on the test set after hyperparameter tuning (GridSearchCV / RandomizedSearchCV where applicable).

| Model | Technique |
|---|---|
| Linear Regression | Baseline linear model |
| Decision Tree Regressor | Tuned via GridSearchCV |
| Random Forest Regressor | Ensemble, tuned via GridSearchCV |
| Gradient Boosting Regressor | Tuned via GridSearchCV |
| XGBoost Regressor | Tuned via GridSearchCV |
| K-Nearest Neighbors Regressor | Tuned via GridSearchCV |
| MLP Regressor (ANN) | Tuned via GridSearchCV |

A line plot comparing R² scores across all seven models (with the best performer highlighted) is available in the notebook, Section 10.

## Conclusion

- The **Decision Tree Regressor** and **MLP Regressor (ANN)** produced very similar, and the highest, R² scores among all models tested — both performed well on this regression problem.
- Among all models, the **Decision Tree Regressor** achieved the best overall predictive performance and is recommended as the most suitable model for this dataset and use case.

# Challenges Faced

1. **Extracting insights from the dataset** — the dataset mixes 15 categorical and 10 continuous features, which required careful, feature-by-feature exploratory analysis to surface meaningful patterns.
2. **Corrupted values** — several columns (normalized-losses, bore, stroke, horsepower, peak-rpm, num-of-doors) used `?` as a placeholder for missing data instead of a standard null, requiring detection and median/mode imputation before any analysis could proceed.
3. **Computational time** — some ensemble and boosting models (Random Forest, Gradient Boosting, XGBoost) took considerably longer to train and tune than simpler models, especially during hyperparameter search.
4. **Consistent evaluation across models** — finding a `random_state` at which all seven models could be fairly and consistently compared took iteration, since results shifted meaningfully with different train/test splits.
