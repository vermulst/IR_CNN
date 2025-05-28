from rdkit import Chem

class FunctionalGroupIdentifier:
    def __init__(self, smarts_dict):
        self.group_names = list(smarts_dict.keys())
        self.smarts_mols = {}
        for name, patt in smarts_dict.items():
            mol = Chem.MolFromSmarts(patt)
            if mol is None:
                raise ValueError(f"Invalid SMARTS pattern for {name}: {patt}")
            self.smarts_mols[name] = mol

    def encode(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES: {smiles}")
                return None
            return [
                int(bool(mol.GetSubstructMatches(self.smarts_mols[name])))
                for name in self.group_names
            ]
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None