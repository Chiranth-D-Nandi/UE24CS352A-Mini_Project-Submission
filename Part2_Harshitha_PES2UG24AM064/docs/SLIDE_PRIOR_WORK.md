# Slide-ready positioning

## Slide: Key findings from the existing HitPredict paper

- **Problem:** Predict whether a song will appear on the Billboard Hot 100 using **nine Spotify audio descriptors** plus a binary **artist score** indicating whether the artist had a previous Hot 100 hit.
- **Dataset:** Approximately 4,000 balanced hit/non-hit songs. Candidate non-hits came from the Million Song Dataset, while hits came from Billboard; Spotify API audio features were then collected for the songs.
- **Sampling-bias limitation:** Because the two classes originated from different source pools, systematic source or selection differences may be learned as if they were differences between hits and non-hits.
- **Evaluation limitation:** Supervised models used a random 75% training / 25% validation split and **no independent test set**. A validation set guides model and hyperparameter choices; a test set is held aside until all choices are complete to estimate final generalization without selection bias.
- **Reported validation results:** Logistic Regression **75.9%**; Gaussian Discriminant Analysis **73.7%**; Neural Network **76.5%**.
- **GDA in one sentence:** Gaussian Discriminant Analysis is a generative classifier that models each class’s feature distribution as Gaussian and uses Bayes’ rule to classify a new example.
- **Neural-network correction:** It was **one hidden layer with six sigmoid units**, not a six-layer network.
- **Overfitting result:** An unrestricted Decision Tree reached 100% training accuracy but only **51.5% validation accuracy**; depth control and bagging/random-forest-style methods were explored to reduce overfitting.
- **Feature findings:** Artist score was most influential for Logistic Regression, while danceability was most influential for the Neural Network. The paper also observed holiday and time-period drift.

## Slide: How our project differs

- **Positioning:** Methodological extension/reimplementation of the same problem—not an exact reproduction of the paper’s experiment or its reported accuracies.
- Constructs labels by conservatively joining a large static Spotify feature catalogue to a licensed historical Billboard archive.
- Uses exact, auditable title/artist linkage and excludes Spotify popularity and chart-outcome variables.
- Keeps natural class prevalence instead of forcing a balanced evaluation dataset.
- Uses chronological training, validation, calibration, and untouched test periods rather than a random 75/25 split.
- Selects using PR-AUC, calibrates probabilities, reports threshold metrics and confidence intervals, and exports a reproducible model artifact and demo.
- **Current status:** pipeline complete, but real metrics remain pending until the real Spotify CSV is supplied and the final run is executed.
