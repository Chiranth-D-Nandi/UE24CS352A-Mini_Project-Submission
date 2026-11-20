import numpy as np
import pandas as pd

from hitpredict.metrics import (
    balanced_evaluation_subset,
    choose_f1_threshold,
    classification_metrics,
)


def test_metrics_and_balanced_reporting_subset():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.9])
    threshold, _ = choose_f1_threshold(y, p)
    metrics = classification_metrics(y, p, threshold)
    assert metrics["average_precision"] is not None
    frame = pd.DataFrame({"billboard_hit": [0] * 20 + [1] * 4})
    balanced = balanced_evaluation_subset(frame, "billboard_hit", 1, 42)
    assert balanced["billboard_hit"].value_counts().to_dict() == {0: 4, 1: 4}
