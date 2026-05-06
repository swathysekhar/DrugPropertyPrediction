import torch
from rdkit import Chem


FUNCTIONAL_GROUPS = {
    "alcohol": "[OX2H]",
    "phenol": "c[OX2H]",
    "amine_primary": "[NX3;H2]",
    "amine_secondary": "[NX3;H1]",
    "amine_tertiary": "[NX3;H0]",
    "amide": "C(=O)N",
    "carboxylic_acid": "C(=O)[OH]",
    "ester": "C(=O)O[#6]",
    "ether": "[OD2]([#6])[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "nitro": "[NX3](=O)=O",
    "halide": "[F,Cl,Br,I]",
    "thiol": "[SX2H]",
    "thioether": "[SX2]([#6])[#6]",
    "sulfonamide": "S(=O)(=O)N",
    "sulfone": "S(=O)(=O)[#6]",
    "phosphate": "P(=O)(O)(O)",
    "alkene": "C=C",
    "alkyne": "C#C",
    "aromatic": "a",
    "benzene_ring": "c1ccccc1",
    "nitrile": "C#N",
}


FG_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in FUNCTIONAL_GROUPS.items()
}


def extract_functional_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return torch.zeros(len(FG_PATTERNS), dtype=torch.float)

    fg_vector = [
        int(mol.HasSubstructMatch(pattern))
        for pattern in FG_PATTERNS.values()
    ]

    return torch.tensor(fg_vector, dtype=torch.float)


def get_functional_group_names():
    return list(FUNCTIONAL_GROUPS.keys())


def get_functional_group_dim():
    return len(FUNCTIONAL_GROUPS)