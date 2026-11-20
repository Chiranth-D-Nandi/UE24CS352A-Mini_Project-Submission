# Implementation and Review Roadmap

The code for phases 0–13 is present; gates that require the real Spotify file/results remain open.

| Phase | Objective / exact command | Outputs | Gate |
|---|---|---|---|
| 0 Understand | Read uploaded assignment/paper/poster | research notes, assignment mapping | no invented constraint |
| 1 Research | Firecrawl dataset/literature/API/deployment evidence | decision log | sources change design |
| 2 Acquire | `hitpredict acquire` + manual Spotify placement | raw files, hashes | schema/licence reviewed |
| 3 Audit source | `hitpredict prepare` | labels, match/unmatched audit | manually inspect linkage |
| 4 Prepare | same command: clean, dedupe, split | modelling CSV | no duplicate entity; both classes/split |
| 5 Baselines | `hitpredict train --profile quick` | dummy/LR/DT records | beat prevalence baseline |
| 6 Candidates | same: RF/HistGB/MLP | validation comparison | no test access |
| 7 Experiments | optional `--profile full` | limited variants | meaningful validation gain |
| 8 Select | automated PR-AUC + parsimony rule | selection JSON | rule applied unchanged |
| 9 Final fit | automated pre-2017 refit + 2017 calibration | joblib artifact | calibrated probabilities valid range |
| 10 Errors | automatic final test analysis | FP/FN/segments/CIs/plots | test reveal count 1 |
| 11 Application | `streamlit run app/app.py` | reviewer UI | valid/invalid inputs work |
| 12 Testing | `pytest -q` | passing suite | no failed contract |
| 13 Deployment | local first; optional Community Cloud | reachable demo | cold start/artifact verified |
| 14 Documentation | use real generated values only | README + PDF | no placeholder called result |
| 15 Presentation | teammate slide deck + rehearsal | slides/demo script | both explain methodology |
| 16 Final audit | run checklist below | submission bundle | every official item mapped |

## Final audit commands

```bash
pytest -q
hitpredict audit --config configs/default.yaml
python -m streamlit run app/app.py
```

Then inspect:

- source hashes and licence notes;
- match coverage and a manual match sample;
- no duplicate `song_group`;
- split year boundaries;
- model comparison and failed experiments;
- one test reveal, CIs, calibration, FP/FN examples;
- secrets/raw data/artifacts appropriate for repository/deployment;
- README from a clean environment;
- truthful contribution record;
- PDF page limit and all official deliverables.
