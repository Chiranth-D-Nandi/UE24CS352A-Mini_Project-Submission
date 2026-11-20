# Model Selection Report Template

Populate this only from generated files. Do not paste expected values as results.

## Evaluation contract

- Primary: validation PR-AUC, natural prevalence.
- Tie rule: within 0.005 PR-AUC choose lower declared complexity, then fewer features, then lower log loss.
- Test: 2018–2020, opened once after selection/calibration/threshold.

## Candidates

Import rows from `reports/metrics/model_comparison_validation.csv`:

| Model | Feature set | PR-AUC | F1 | ROC-AUC | Log loss | Fit seconds | Strength | Weakness |
|---|---|---:|---:|---:|---:|---:|---|---|
| Dummy | | | | | | | prevalence floor | no signal |
| LR | | | | | | | interpretable/cheap | linear |
| Shallow DT | | | | | | | nonlinear rules | unstable |
| RF | | | | | | | robust interactions | size/extrapolation |
| HistGB | | | | | | | strong tabular ranking | tuning sensitivity |
| MLP | | | | | | | smooth interactions | variance/complexity |

## Final selection

- **Selected design:** [generated candidate + feature set]
- **Why:** [metric and tie-rule evidence]
- **Why alternatives rejected:** [concrete validation/cost/calibration evidence]
- **Calibration/threshold:** [2017 metrics and threshold]
- **Untouched test:** [natural-prevalence metrics + 95% intervals]
- **Balanced comparison-only:** [clearly label as different prevalence]
- **Deployment suitability:** [artifact size and measured latency]

## Bad/negative results

List any model that did not beat LR/dummy, any artist-history ablation that failed, and any expanded variant that failed to justify cost. These are valid findings.
