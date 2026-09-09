# Multi-target QSAR / bioactivity classification

## 1. Problem

Multi-target QSAR (quantitative structure-activity relationship): given only
a molecule's 2D structure, predict whether it is active against each of
eleven biological targets. This is the everyday triage problem in early
drug discovery — experimental screening across a target panel is
expensive, so a model that ranks compounds per target decides what
actually gets tested. Labels are three-valued (`1` = active, `0` = unknown /
not measured, `-1` = inactive), which is what real screening matrices look
like: sparse and incomplete, not a clean binary split.

**The eleven targets were never disclosed by the challenge.** They appear
only as `task1`…`task11`; no assay, protein, or endpoint identity was ever
given, and this README does not speculate about what they might be. The
anonymity rules out any use of target-specific priors — the model is
purely structure-driven, which is the honest framing of what was, and was
not, possible here.

## 2. Approach

Molecules are standardised with RDKit (`Cleanup` + `FragmentParent`), then
represented by ~200 2D descriptors (standardised with a `StandardScaler`
fit on the training split) concatenated with a 2048-bit, radius-2 Morgan
fingerprint. Two strategies are compared on identical features:

- **Per-target**: eleven independent LightGBM (or RandomForest) classifiers,
  one per target.
- **Multi-output**: a single `MultiOutputClassifier` wrapping one LightGBM
  (or RandomForest).

Both use `RandomizedSearchCV` (12 draws, 3-fold) for hyperparameters, then
5-fold `KFold` for the reported internal scores, with
`class_weight='balanced'` to offset label imbalance. Test predictions are
the mean over the five fold models plus a final model fit on all training
data. RandomForest was run through the same harness as a comparison point.

## 3. Results

**Evaluation setting:** public test set via the challenge's leaderboard
server; per-task ROC AUC over the eleven tasks, reported as the arithmetic
mean. Single submission each, one seed (42), no repeats. Full table in
[`results/leaderboard.md`](results/leaderboard.md).

| Strategy | Mean ROC AUC |
|---|---|
| **Per-target LightGBM (11 models)** | **0.668** |
| Single multi-output LightGBM | 0.663 |

Per-task AUC, per-target LightGBM:

| Task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.775 | 0.525 | 0.900 | 0.506 | 0.876 | 0.776 | 0.519 | 0.576 | 0.864 | 0.647 | **0.383** |

The mean (0.668) hides a wide spread. The eleven targets separate into
three groups: four are genuinely learnable from 2D structure alone (tasks
3, 5, 9, 1 at 0.775–0.900), four are at or near chance (tasks 2, 4, 7, 8 at
0.506–0.576), and task 11 sits at 0.383 — below chance. Below-chance AUC is
worth flagging rather than explaining away: it is consistent with either a
label-orientation issue or noise from a small number of positives, but
**the number of active labels per task is not recorded anywhere**, so the
mechanism can't be determined from what survives. I'm stating it that way
rather than asserting an inverted ranking.

The 0.005 gap between the two strategies (0.668 vs. 0.663) is well inside
what a single seed, single split can explain. The honest reading is that
per-target and multi-output performed about the same here, not that
per-target strategy won.

No internal cross-validation figures are reported: this run's
`training_auc_log.txt` did not survive. Only the leaderboard numbers above
are quoted, and they are the unbiased ones — see Limitations for why the
internal CV numbers would have been optimistic anyway.

## 4. Reproduction

```
pip install -r requirements.txt
```

The original challenge data is not available (see
[`data/README.md`](data/README.md)), so reproduction runs on synthetic
data:

```
python scripts/make_example_data.py
python scripts/prepare_data.py
python scripts/train.py
python scripts/predict.py
```

This exercises the full pipeline end-to-end and **will not reproduce the
reported AUCs** — nothing can, without the original dataset. Use
`scripts/train.py --models lgbm --strategy per-target` (or any of `rf`,
`multi-output`, `both`) to run a single path instead of all four
combinations. Original runtime and hardware are **not recorded**; on the
real data the hyperparameter search is the expensive part (12 draws × 3
folds, repeated per target for the per-target strategy).

## 5. Repo structure

```
README.md
LICENSE                     MIT
THIRD_PARTY.md              all code is mine; third-party library licences only
requirements.txt            reconstructed + pinned (original versions not recorded)
.gitignore
src/qsar/
  __init__.py
  features.py                standardisation, descriptors, Morgan fingerprint
  data.py                    load / split / scale
  metrics.py                 the two AUC scorers
  models.py                  hyperparameter grids, the two training strategies
  predict.py                 the two prediction writers
scripts/
  make_example_data.py       synthetic data in the documented schema — added at
                              publication time, NOT part of the original submission
  prepare_data.py            argparse entry point
  train.py                   argparse entry point
  predict.py                 argparse entry point
data/README.md               schema + provenance, no data
results/
  leaderboard.md             mean + per-task AUCs
```

## 6. Limitations and what I would do next

Written as specifics, each traceable to a line of code, and none of them
fixed — this repo is a record of what was actually submitted, not a
cleaned-up version of it.

- **Submissions average hard class predictions, not probabilities.**
  `predict_multioutput` and `predict_per_target` call `model.predict(X)`
  and take the mean over six models. Averaging labels drawn from
  `{-1, 0, 1}` produces at most about thirteen distinct values, so the
  submitted ranking is heavily tied, and ROC AUC degrades with tie density.
  **This is not a hypothesis:** the challenge's own `sample_submission.csv`
  is populated with continuous scores in `[0, 1]`, so a ranking score was
  what the format called for. Switching to `predict_proba` is the single
  highest-value change here, and is likely worth more than any
  hyperparameter adjustment. Identifying the format mismatch in my own
  submission is a stronger signal than a marginally higher score would
  have been.
- **Unknown labels (`0`) are not masked.** Each task is fitted as a
  three-class problem and scored one-vs-rest over `labels=[-1, 0, 1]`, so
  the model spends capacity predicting "not measured" — a property of the
  assay panel, not of the molecule. Standard practice is to mask unmeasured
  entries out of the loss.
- **Hyperparameters are searched on the full training set**
  (`search.fit(X, y)`) before the same data is used for 5-fold CV, so the
  internal fold scores are optimistically biased. The leaderboard number is
  unaffected by this.
- **`class_weight='balanced'` is applied to the CV and final models but not
  inside `RandomizedSearchCV`**, so the search selects hyperparameters
  under a different objective than the models that ship.
- **The reported internal average mixes five CV-fold scores with one
  holdout score** in a single mean — two different evaluation settings
  averaged together.
- **The LightGBM grid passes `feature_fraction` and `bagging_fraction`**
  through the scikit-learn wrapper; `bagging_fraction` has no effect
  without `bagging_freq > 0`, so that part of the grid was probably inert.
  Worth verifying rather than asserting.
- **No scaffold split.** Random splitting of molecules inflates apparent
  performance because close analogues land on both sides of the split; a
  Murcko-scaffold split is the standard correction.
- **Test-molecule preservation, traced rather than assumed** (the challenge
  explicitly warned that standardisation must not drop test molecules):
  `get_descriptors` (`src/qsar/features.py`) calls
  `standardize_mol(Chem.MolFromSmiles(smiles))` before any error handling.
  If a SMILES fails to parse, `Chem.MolFromSmiles` returns `None`, and
  `rdMolStandardize.Cleanup(None)` raises an **uncaught** `ValueError`.
  That exception propagates out of `pandas.Series.apply` and halts the
  whole preprocessing run — verified empirically (RDKit 2023.09,
  pandas 2.1). So on the evidence available, this pipeline does not
  silently drop unparseable test molecules; it crashes instead. Every
  molecule in a run that produced output therefore parsed successfully. A
  separate, narrower risk (a molecule that parses but fails
  descriptor/fingerprint calculation, caught by the second `try`/`except`
  in `get_descriptors`) was not observed in testing but is not ruled out by
  inspection alone.
- Next: probability outputs, masked multi-task loss, scaffold-split
  validation, and a look at whether task 11's sub-chance AUC is a
  label-orientation problem or a genuine inverted signal.

## 7. Provenance

University challenge submission, AI in Life Sciences, summer semester
2025. Solo work. Code dated 2025-04-07; submissions 2025-04-07 20:46 and
20:47 (server timestamps, one minute apart — consistent with uploading two
files from one run). All code is my own; no third-party or course-provided
source is included (the challenge supplied only data and the leaderboard).

The mapping from the two code paths (`cross_val_single_model` /
`train_per_target`) to the two leaderboard entries quoted above is
**inferred** from the submission descriptions and the code's output
filenames — the submitted prediction CSVs did not survive, and the
challenge server had no downloadable submission history left as of
2026-09-08. Two further submissions (RandomForest, through the same two
code paths) have no surviving leaderboard record and are not discussed
here.

`scripts/make_example_data.py` was written at publication time (2026) to
make the pipeline runnable without the original data; everything else is
as submitted, modulo refactoring for legibility (module split, docstrings,
CLI arguments, the `predict_ensemble_rf` → `predict_multioutput` rename,
explicit `set_seed(42)`). No hyperparameter, feature, or modelling choice
was changed from the original submission. **No grade is stated anywhere in
this repository.**
