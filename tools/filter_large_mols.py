
from fastvs.core.io import load_smiles_file
import os, sys
from rdkit import Chem


def filter_molecules(smiles_data, heavy_atoms_threshold=20):
   filtered_smiles = []
   for i, (smiles, name) in enumerate(smiles_data):
        if i % 10000 == 0:
            print(f'[INFO] progress loading molecules {i}')
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumHeavyAtoms() <= heavy_atoms_threshold:
            filtered_smiles.append([smiles, name])
   
   return filtered_smiles


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: filter_large_mols.py input.smi 20')
        sys.exit(0)

    # Load or create your dataset of molecules and their labels
    # For this example, we'll assume you have a CSV file with columns 'smiles' and 'label'
    print("[INFO] load molecules from input file.")
    smiles_data = list(load_smiles_file(sys.argv[1]).values())

    # Filter the molecules with heavy atoms number lower than 20
    filtered_smiles = filter_molecules(smiles_data, heavy_atoms_threshold=int(sys.argv[2]))

    print("[INFO] write molecules into output file.")
    with open(f'filter_ac-lt-{sys.argv[2]}_{os.path.basename(sys.argv[1])}', 'w') as tofile:
        for (s, n) in filtered_smiles:
            tofile.write(f'{s} {n}\n')
    