# Leaderboard results

**Evaluation setting:** public test set, challenge-server leaderboard. Per-task
ROC AUC over eleven anonymised targets, reported as the arithmetic mean.
Single submission each, one seed (42), no repeats. Source: challenge
leaderboard entries recovered at Gate 2 (2026-09-08); the submitted
prediction CSVs themselves did not survive, so the mapping from these two
leaderboard entries to the two code paths below is **inferred** from the
submission descriptions and the code's output filenames, not confirmed
against a preserved artefact. See the main README's Provenance section.

| Strategy | Code path | Output file | Mean ROC AUC |
|---|---|---|---|
| **Per-target LightGBM (11 models)** | `train_per_target` → `predict_per_target` | `predictions_per_target_LGBM_test.csv` | **0.668** |
| Single multi-output LightGBM | `cross_val_single_model` → `predict_multioutput` | `predictions_single_LGBM_test.csv` | 0.663 |

Two further submissions used RandomForest through the same two code paths;
no leaderboard record of those survives.

## Per-task AUC, per-target LightGBM

| Task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.775 | 0.525 | 0.900 | 0.506 | 0.876 | 0.776 | 0.519 | 0.576 | 0.864 | 0.647 | **0.383** |

Mean of the eleven values above: 0.6679, rounding to the stated 0.668.

No internal cross-validation figures are reported here: the run's
`training_auc_log.txt` did not survive, so only the unbiased leaderboard
numbers above are quoted. See the main README's Results and Limitations
sections for discussion, in particular of task 11's below-chance AUC and
why the per-target vs. multi-output gap (0.668 vs. 0.663) is not
interpreted as one strategy beating the other.
