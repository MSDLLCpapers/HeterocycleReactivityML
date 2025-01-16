import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

def feature_importance(model=None, X=None):
    """
    Calculates permutation feature importance from a fitted model.

    Parameters:
    - model: fitted model (with feature_importances_ attribute).
    - X: feature dataframe used for the model.

    Returns:
    - feature_list_ordered: a sorted series of feature importances.
    """
    if model is None or X is None:
        raise ValueError("Model and feature dataframe (X) must be provided.")

    # Get feature importances and standard deviation
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    feature_names = X
    forest_importances = pd.Series(importances, index=feature_names)

    # Sort features by importance
    feature_list_ordered = forest_importances.sort_values(ascending=False)
    return feature_list_ordered

def visualize_feature(feature=None, smi=None):
    """
    Visualizes a specific feature in a molecule's fingerprint.

    Parameters:
    - feature: feature to visualize (string indicating the bit in fingerprints).
    - smi: SMILES representation of molecule.

    Returns:
    - None: Displays visual representation of fingerprint bit.
    """
    if feature is None or smi is None:
        raise ValueError("Feature and SMILES string must be provided.")

    bit_no = int(feature[3:])  # Extract bit number from feature string

    mol = Chem.MolFromSmiles(smi)
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bi)

    # Draw specified bit of fingerprints
    img = Draw.DrawMorganBit(mol, bit_no, bi)
    return img  # Return image
