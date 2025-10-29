

import os, sys
import numpy as np
import shutil
from fastvs.core.utils import run_command, make_temp_dpath, encode_string
from rdkit import Chem
from rdkit.Chem import AllChem

def nyan_fingerprint(smiles_list: list, configs: dict) -> dict: 
    """
    Arguments
    ---------
    smiles_list: list of smiles
        The smiles for encoding
    configs: dict
        The configuration paramters
    """
    out_dpath = make_temp_dpath()
    inp_fpath = os.path.join(out_dpath, 'nyan_inp.csv')
    out_fpath = os.path.join(out_dpath, 'nyan_out.csv')

    results = {}

    # make input data
    with open(inp_fpath, 'w') as tofile:
        for i, smi in enumerate(smiles_list):
            tofile.write(f'{smi} MOL{i}\n')
            results[smi] = []

    # make commands
    cmd = f'{configs["nyan"]["python"]} {configs["nyan"]["script"]} {inp_fpath} {out_fpath}'    
    run_command(cmd, verbose=True)

    # read results
    with open(out_fpath) as lines:
        for l in lines:
            smi = l.split()[0]
            dat = [float(x) for x in l.split()[1:]]

            results[smi] = dat
    
    # clean up
    shutil.rmtree(out_dpath)

    return results


def morgan_fingerprint(smiles_list, mode="bits") -> dict:

    results = {}
    for smi in smiles_list:
        results[smi] = []

        try:
            mol = Chem.MolFromSmiles(smi)
            if mode != "bits":
                results[smi] = AllChem.GetMorganFingerprint(mol, 2)
            else:
                results[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        except:
            print(f'[WARNING] failed in molecule ({smi}) parsing and fingerprint')

    # shape (N, 512)
    return results


class FeaturizeMolecules(object):

    def __init__(self, molecules_dict, fp_types = ['nyan', 'morgan'], configs=None) -> None:
        
        self.molecules_dict = molecules_dict
        #self.fingerprints = None 
        self.fp_types = fp_types
        self.configs = configs
        self.hashid_list  = list(self.molecules_dict.keys())
        #print(self.hashid_list)
        self.smiles_list  = [self.molecules_dict[x].smiles for x in self.hashid_list]
        self.smi_hid_dict = {k:v for k,v in zip(self.smiles_list, self.hashid_list)}
    
    def _prepare_nyan_fp(self):
        if len(self.smiles_list) < 1:
            return {}

        # self.nyan_fp is a dict {smiles: fp}
        return nyan_fingerprint(self.smiles_list, configs=self.configs)

    def _prepare_morgan_fp(self):
        if len(self.smiles_list) < 1:
            return {}
            
        return morgan_fingerprint(self.smiles_list)

    def get_fp(self, fp_data_prev):
        fp_data = {k:[] for k in self.hashid_list}
        for hid in list(fp_data_prev.keys()):
            fp_data[hid] = fp_data_prev[hid]
            try:
                self.smiles_list.remove(self.molecules_dict[hid].smiles)
            except KeyError:
                pass
        
        # shape 64
        if 'nyan' in self.fp_types:
            _fp = self._prepare_nyan_fp()

            for smiles in list(_fp.keys()):
                hashid = self.smi_hid_dict[smiles]
                fp_data[hashid] += _fp[smiles]
        
        # shape 512
        if 'morgan' in self.fp_types:
            _fp = self._prepare_morgan_fp()
            #print(_fp.items())

            for smiles in list(_fp.keys()):
                hashid = self.smi_hid_dict[smiles]
                fp_data[hashid] += _fp[smiles]

        # max value len
        max_value_len = max([len(x) for x in list(fp_data.values())[:100]])
        #print("max value len", max_value_len)
        
        return {key: value for key, value in fp_data.items() if len(value) == max_value_len}
    

if __name__ == '__main__':

    from fastvs.core.io import load_molecule_dataset
    from fastvs.core.utils import load_configs, encode_string

    configs = load_configs('../fastvs/data/configs.json')

    res = load_molecule_dataset('output/scores.npy')
    ids = list(res.keys())
    
    for i in range(10):
        print(res[ids[i]], res[ids[i]].name, res[ids[i]].vinascore, res[ids[i]].smiles, res[ids[i]].docked_)

    fp = FeaturizeMolecules(res, fp_types=['morgan', ], configs=configs)

    data = fp.get_fp()
    ids = list(data.keys())
    print(data[ids[0]])
