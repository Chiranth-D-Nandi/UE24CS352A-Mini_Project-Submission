# Experiment Ledger Policy

The executable ledger is generated at `reports/experiments/experiment_ledger.csv`. One row is written only after a model actually fits and produces validation predictions.

## Record schema

```text
Experiment ID:
Date (UTC):
Question:
Hypothesis:
Dataset and source hashes:
Fit period:
Evaluation period:
Features:
Model and hyperparameters:
Training rows / class ratio:
Metric (predeclared):
Actual result:
Baseline comparison:
Fit time:
Interpretation:
Decision:
Next experiment:
```

## Experiment sequence

| IDs | Purpose | Decision gate |
|---|---|---|
| E01–E06 per feature set | dummy, LR, shallow DT, RF, HistGB, MLP fixed candidates | compare 2015–16 PR-AUC |
| expanded profile only | limited predeclared variants | improvement must exceed noise/tolerance |
| feature-set pair | audio only vs strict artist history | context retained only if validation gain is meaningful |
| final | selected design, 2017 calibration/threshold, 2018–20 test | reveal once; no redesign from test |

A failed model remains in the ledger. Expected performance is never written into an actual-result field.
