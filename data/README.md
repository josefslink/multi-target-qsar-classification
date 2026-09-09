# Data

**The challenge dataset is not included in this repository and is not
redistributable.** It was provided by a university course (AI in Life
Sciences, summer semester 2025) for a bounded challenge exercise; I no
longer have a copy, and course permission to publish covers the code and
results, not the dataset. This directory documents the schema so the code
is legible without the data, and ships a script
(`scripts/make_example_data.py`) that generates structurally-equivalent
synthetic files for a runnable demo.

## Schema (inferred from the code, `train.py:16-19` and `data_prep.py`)

| File | Columns | Notes |
|---|---|---|
| `data_train.csv` | index, `smiles`, then 11 label columns | Read with `header=[0], index_col=[0]`. Labels: `1` = active, `0` = unknown/not measured, `-1` = inactive. The code takes the first 11 columns after featurisation as `y` and the rest as `X`. Original column names for the 11 label columns are **not recorded** — the code addresses them positionally, never by name. |
| `smiles_test.csv` | index, `smiles` | Same read convention as above. |
| `sample_submission.csv` | unnamed index, `task1`…`task11` | Format reference only — **never read by the code**. Values are continuous scores in `[0, 1]` (verbatim from the challenge page), i.e. the expected output is a per-task ranking score, not a class label. See the README's Limitations section for what this means for this submission. |

## Feature matrix

Per molecule: all RDKit 2D descriptors from `Descriptors.descList` (~200,
standardised with a `StandardScaler` fit on the training split), plus a
2048-bit Morgan fingerprint at radius 2 (left unstandardised). Molecules
are cleaned with `rdMolStandardize.Cleanup` then reduced to their
`FragmentParent` before featurisation. See `src/qsar/features.py`.

## What is not recorded

- **Training set row count.** Not present anywhere in the surviving code
  or artefact; the challenge page's continued availability was checked at
  publication time (2026-09) and the row count could not be recovered from
  it either.
- **The eleven targets' identities.** The challenge anonymised them
  deliberately — they are `task1`…`task11` and nothing more. No assay,
  protein, or endpoint identity was ever disclosed to participants.
- **Number of active (`1`) labels per task.** Relevant to interpreting the
  below-chance AUC on task 11 (see the main README); not recoverable from
  what survives.

## Acquisition

There is no acquisition path for the original data — it was distributed
only to registered course participants for the duration of the challenge
and is not otherwise published. To run this pipeline end-to-end without it,
use the synthetic data generator:

```
python scripts/make_example_data.py
```

This writes `data_train.csv` and `smiles_test.csv` into this directory in
the schema above, using public SMILES and random labels. It is clearly
not a substitute for the real dataset and will not reproduce the reported
scores — see the top-level README.
