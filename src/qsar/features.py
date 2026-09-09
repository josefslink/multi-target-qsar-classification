"""Molecule standardisation and featurisation.

Ported from the original submission's ``data_prep.py``. Two dead imports
present in the original (``argparse``, ``rdkit.Chem.AllChem`` — neither was
referenced anywhere in that file) are dropped here. Every featurisation
parameter is otherwise unchanged: RDKit's full ``Descriptors.descList``
(~200 2D descriptors) plus a 2048-bit, radius-2 Morgan fingerprint, on a
molecule reduced to its ``FragmentParent`` after ``rdMolStandardize.Cleanup``.
"""
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

#: Names of every descriptor in RDKit's ``Descriptors.descList``, in order.
DESCRIPTOR_NAMES = [name for name, _ in Chem.Descriptors.descList]

#: Morgan fingerprint bit width. Unchanged from the original submission.
FP_SIZE = 2048


def standardize_mol(mol):
    """Clean ``mol`` and reduce it to its largest-fragment parent.

    Runs ``rdMolStandardize.Cleanup`` followed by ``FragmentParent``,
    unchanged from the original submission's ``standardize_mol``.

    Note on test-molecule preservation (see README Limitations): if ``mol``
    is ``None`` (e.g. produced by ``Chem.MolFromSmiles`` failing to parse a
    SMILES string), ``rdMolStandardize.Cleanup`` raises an uncaught
    ``ValueError``. That exception is not caught anywhere in this module or
    in ``get_descriptors`` below, so an unparseable molecule crashes the
    whole preprocessing run rather than being silently skipped.
    """
    clean = rdMolStandardize.Cleanup(mol)
    parent = rdMolStandardize.FragmentParent(clean)
    return parent


def get_descriptors(smiles: str):
    """Compute the feature vector for one SMILES string.

    Returns a ``pandas.Series`` of RDKit 2D descriptors
    (``DESCRIPTOR_NAMES``) concatenated with a 2048-bit radius-2 Morgan
    fingerprint (columns ``FP_0`` .. ``FP_2047``), or ``None`` if descriptor
    or fingerprint calculation raises after a successful parse and
    standardisation (caught and printed, matching the original).

    Unchanged from the original submission's ``get_descriptors``, including
    its error handling: an unparseable SMILES is *not* caught here (see
    ``standardize_mol``) and propagates as an uncaught exception.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = standardize_mol(mol)

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"sanitization failed for {smiles}, Error: {e}")

    try:
        desc_series = pd.Series(CalcMolDescriptors(mol), index=DESCRIPTOR_NAMES)

        morgan_gen = GetMorganGenerator(radius=2, fpSize=FP_SIZE)
        fp = morgan_gen.GetFingerprint(mol)
        fp_array = np.zeros((FP_SIZE), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        fp_series = pd.Series(fp_array, index=[f"FP_{i}" for i in range(FP_SIZE)])

        return pd.concat([desc_series, fp_series])

    except Exception as e:
        print(f"Descriptor calculation failed for SMILES: {smiles}, Error: {e}")
