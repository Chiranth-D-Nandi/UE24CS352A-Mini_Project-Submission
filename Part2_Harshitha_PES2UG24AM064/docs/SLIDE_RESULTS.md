# Slide-ready results

## Slide: Dataset and Evaluation

- **Data source:** 1.2M Kaggle Spotify records linked against UT Data's Billboard Hot 100 archive.
- **Linkage strategy:** Exact string matching after strict normalization (no fuzzy matching) to ensure zero false-positive labels.
- **Final cohort:** 1,030,990 valid song entities.
- **True hits:** 2,881 unique Billboard hits (0.28% natural prevalence).
- **Temporal split:**
  - Training: 1990–2014 (736,825 rows, 1,711 hits)
  - Validation: 2015–2016 (83,834 rows, 193 hits)
  - Calibration: 2017 (44,419 rows, 164 hits)
  - Test: 2018–2020 (165,912 rows, 813 hits)

## Slide: Model Selection

- **Candidates tested:** Logistic Regression, Decision Tree, Random Forest, Histogram Gradient Boosting, Multilayer Perceptron.
- **Winner:** Random Forest (15 features: audio + prior artist hits).
- **Hyperparameters:** `max_depth` = 14, `n_estimators` = 250, `min_samples_leaf` = 5.
- **Why Random Forest:** Bagging captured non-linear interactions without the catastrophic overfitting observed in unrestricted single trees. It had the highest average precision (PR-AUC) on the 2015–2016 validation period (0.096).
- **Calibration:** Sigmoid calibration was applied using the 2017 data to correct for the negative-undersampled training prior, allowing the model to produce realistic probabilities.

## Slide: Final Results (2018–2020 Test Set)

- **Performance metric:** PR-AUC was **0.101** on the test set, substantially beating random guessing (0.005) but confirming that audio features alone are not deterministically predictive of hit status.
- **Operating point:** F1 score is 0.171, Recall is 0.306, and Precision is 0.119.
- **Artist history effect:** Removing prior artist hits from the features dropped the model's F1 score and recall to 0 for artists without a prior hit. Prior success is a very strong proxy for future success.
- **Conclusion:** While Spotify audio descriptors show statistically significant association with hit probabilities, they lack the causal context (marketing, social momentum, changing tastes) necessary to reliably predict chart appearance in a real-world, natural-prevalence scenario.
