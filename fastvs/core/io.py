

import os, sys
from fastvs.core.utils import run_command, make_temp_dpath, encode_string
from fastvs.core.data import MoleculeData
import shutil
import random
import numpy as np


OBABEL_BINARY = "/sugon_store/zhengliangzhen/.conda/envs/sfct/bin/obabel"


def smiles2pdb(smiles, out_dpath, mode="obabel", return_cmd=True):
    os.makedirs(out_dpath, exist_ok=True)

    #if os.path.exists(os.path.join(out_dpath, 'ligand.pdb')):
    #    return "" 

    tmp_fpath = f'{out_dpath}/input.smi'
    #if os.path.exists(tmp_fpath):
    #    return ""

    with open(tmp_fpath, 'w') as tofile:
        tofile.write(f'{smiles} molecule\n')

    if mode == "obabel":
        cmd = f"{OBABEL_BINARY} {tmp_fpath} -O {out_dpath}/ligand.pdb --gen3D --minimize --ff MMFF94 -p 7.4"
    else:
        cmd = ""

    if not return_cmd:
        run_command(cmd)

    return cmd


def load_smiles_file(fpath: str) -> dict:
    with open(fpath) as lines:
        data = [x for x in lines]
        smiles = [x.split()[0] for x in data]
        names = [x.split()[-1] for x in data]
        ids = [encode_string(x) for x in smiles]

    # {"hasid": (smiles, name)}
    molecules =  {k : v for k,v in zip(ids,zip(smiles, names))}

    return molecules


def split_smiles_dict(mols: dict, ratio: float =0.1, seed=1024) -> (dict, dict):
    # Set a fixed random seed
    random.seed(seed)

    mol_ids = list(mols.keys())
    # num moles
    n = int(len(mol_ids) * ratio)

    # select n keys
    random.shuffle(mol_ids)
    skeys = mol_ids[:n]
    #skeys = sorted([random.choice(mol_ids) for _ in range(n)])

    set1 = {k:v for k,v in zip(skeys, [mols[x] for x in skeys])}

    skeys2 = [x for x in mol_ids if x not in skeys]
    set2 = {k:v for k,v in zip(skeys2, [mols[x] for x in skeys2])}

    return set1, set2


def split_smiles_dict_by_num(mols: dict, n: int = 1000, seed=1024) -> (dict, dict):
    # Set a fixed random seed
    random.seed(seed)

    mol_ids = list(mols.keys())
    # num moles
    if n > len(mol_ids):
        n = len(mol_ids)

    # select n keys
    random.shuffle(mol_ids)
    skeys = mol_ids[:n] #sorted([random.choice(mol_ids) for _ in range(n)])

    set1 = {k:v for k,v in zip(skeys, [mols[x] for x in skeys])}

    skeys2 = [x for x in mol_ids if x not in skeys]
    set2 = {k:v for k,v in zip(skeys2, [mols[x] for x in skeys2])}

    return set1, set2

def save_molecule_dataset(dataset: list, out_fpath: str):
    
    np.save(out_fpath, dataset)

    return dataset


def load_molecule_dataset(inp_fpath: str) -> list:

    data = np.load(inp_fpath, allow_pickle=True).item()
    
    return data


if __name__ == "__main__":
    import os, sys

    data = load_smiles_file(sys.argv[1])

    keys = list(data.keys())

    m = [MoleculeData(smiles=data[keys[0]][0], name=data[keys[0]][1], hashid=keys[0]), 
         MoleculeData(smiles=data[keys[1]][0], name=data[keys[1]][1], hashid=keys[1])]
    print(m)

    save_molecule_dataset(m, 'out.npy')

    m = load_molecule_dataset('out.npy')
    print('Load data ', m)


    