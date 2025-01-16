import pandas as pd
import numpy as np
from rdkit import DataStructs
import rdkit.Chem as Chem
from rdkit.Chem import AllChem, Descriptors, PandasTools
from rdkit.ML.Descriptors import MoleculeDescriptors


def calc_rdkit_desc(df):
    # Add a molecule column to the DataFrame using the canonical smiles
    PandasTools.AddMoleculeColumnToFrame(df, 'Canonical_Smiles')

    # Calculate RDKit descriptors
    desc_list = [descriptor[0] for descriptor in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)

    # Calculate descriptors for each molecule
    rdkit_desc = [calc.CalcDescriptors(Chem.MolFromSmiles(m)) for m in df.Canonical_Smiles]

    # Create a DataFrame from the calculated descriptors
    rdkit_desc_df = pd.DataFrame(rdkit_desc, columns=desc_list)
    return rdkit_desc_df


def calc_rdkit_fp(df, radius=2, nBits=2048):
    # Convert molecular structures to fingerprint vectors
    mfp2_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits) for mol in df['ROMol']]
    
    # Convert fingerprint vectors to numpy arrays
    FPS = []
    for fp in mfp2_fps:
        arr = np.zeros((nBits,))  # Create an array of zeros with length nBits
        DataStructs.ConvertToNumpyArray(fp, arr)
        FPS.append(arr)
        
    # Create column names for the fingerprint DataFrame
    col_names = [f'bit{x}' for x in range(nBits)]
    rdkit_fp_df = pd.DataFrame(FPS, columns=col_names)
    return rdkit_fp_df