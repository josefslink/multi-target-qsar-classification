#!/usr/bin/env python3
"""Generate synthetic example data in the challenge's documented schema.

**Added at publication time — not part of the original challenge
submission.** The real challenge data (``data_train.csv``, ``smiles_test.csv``)
is not redistributable and is no longer in my possession (see
``data/README.md``). This script produces substitute files with the same
column layout so that ``prepare_data.py`` -> ``train.py`` -> ``predict.py``
can be run end-to-end by a reviewer without source edits. It does **not**
reproduce, and is not intended to reproduce, the leaderboard scores in
``results/leaderboard.md`` — nothing can, without the original dataset. See
the README's Reproduction section.

Molecule source: RDKit's own bundled NCI Open Database sample
(``rdkit/Data/NCI/first_5K.smi``), which ships inside the ``rdkit`` package
and is used in RDKit's test suite — no network access or extra download is
required. If that file is unavailable in a given RDKit install, a small
hardcoded fallback list of well-known public molecules is cycled instead.

Labels (``1`` = active, ``0`` = unknown, ``-1`` = inactive, matching the
challenge's encoding) are assigned uniformly at random per the seed below.
They carry no biological meaning — this is structurally-valid placeholder
data, not a stand-in for the real assay results.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd
from rdkit import RDConfig

from qsar.seeding import SEED, set_seed

N_TASKS = 11

# Small fallback pool if RDKit's bundled NCI sample isn't found (e.g. a
# minimal-data build). All are well-known, public small molecules.
FALLBACK_SMILES = [
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "c1ccccc1",  # benzene
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
    "OCC(O)CO",  # glycerol
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccc2c(c1)ccc(=O)o2",  # coumarin
    "CC(C)NCC(O)c1ccc(O)c(O)c1",  # isoproterenol-like
]


def _load_nci_smiles(max_n: int) -> list[str]:
    """Read up to ``max_n`` SMILES from RDKit's bundled NCI sample file."""
    path = os.path.join(RDConfig.RDDataDir, "NCI", "first_5K.smi")
    smiles = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                smiles.append(parts[0])
            if len(smiles) >= max_n:
                break
    return smiles


def get_example_smiles(n: int) -> list[str]:
    """Return ``n`` public SMILES strings.

    Prefers RDKit's bundled NCI sample; falls back to cycling
    ``FALLBACK_SMILES`` (with repeats) if that file isn't available.
    """
    try:
        pool = _load_nci_smiles(n)
        if len(pool) >= n:
            return pool
    except OSError:
        pass

    pool = []
    while len(pool) < n:
        pool += FALLBACK_SMILES
    return pool[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=300, help="number of synthetic training rows")
    parser.add_argument("--n-test", type=int, default=60, help="number of synthetic test rows")
    parser.add_argument("--out-dir", default="data", help="directory to write data_train.csv / smiles_test.csv")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    all_smiles = get_example_smiles(args.n_train + args.n_test)
    rng.shuffle(all_smiles)
    train_smiles = all_smiles[: args.n_train]
    test_smiles = all_smiles[args.n_train :]

    label_cols = {
        f"task{i + 1}": rng.choices([1, 0, -1], k=args.n_train) for i in range(N_TASKS)
    }
    train_df = pd.DataFrame({"smiles": train_smiles, **label_cols})
    train_df.index.name = ""

    test_df = pd.DataFrame({"smiles": test_smiles})
    test_df.index.name = ""

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "data_train.csv")
    test_path = os.path.join(args.out_dir, "smiles_test.csv")
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    print(f"Wrote {len(train_df)} synthetic training rows to {train_path}")
    print(f"Wrote {len(test_df)} synthetic test rows to {test_path}")


if __name__ == "__main__":
    main()
