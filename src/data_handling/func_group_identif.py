from rdkit import Chem

class FunctionalGroupIdentifier:
    def __init__(self, smarts_dict):
        self.group_names = list(smarts_dict.keys())  # Store the names of the functional groups
        self.smarts_mols = {}  # Dictionary to hold RDKit Mol objects for SMARTS patterns
        self.mol_cache = {}    # Cache to avoid reprocessing the same SMILES

        for name, patt in smarts_dict.items():
            mol = Chem.MolFromSmarts(patt)  # Convert SMARTS pattern to Mol
            if mol is None:
                raise ValueError(f"Invalid SMARTS pattern for {name}: {patt}")
            self.smarts_mols[name] = mol  # Store compiled pattern

    def encode(self, smiles):
        if not smiles:
            return None  # Return None for empty input

        try:
            if smiles in self.mol_cache:
                mol = self.mol_cache[smiles]  # Use cached Mol if available
            else:
                mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to Mol
                if mol is None:
                    print(f"Invalid SMILES: {smiles}")
                    return None
                self.mol_cache[smiles] = mol  # Cache the Mol

            # Check for matches of each functional group and return binary presence list
            return [
                int(bool(mol.GetSubstructMatches(self.smarts_mols[name])))
                for name in self.group_names
            ]

        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None
