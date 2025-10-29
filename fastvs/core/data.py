
from fastvs.core.utils import encode_string

class MoleculeData(object):
    def __init__(self, smiles=None, name=None, hashid=None):

        self.smiles = smiles
        self.name   = name
        self.hashid = hashid

        if hashid is None and self.smiles is not None:
            self.hashid = encode_string(self.smiles)

        self.vinascore = None
        self.sfctscore = None
        self.sfct_weight_ = 0.8

        # predicted scores and their related model
        self.p_sfctscore = None
        self.p_vinascore = None
        self.pvinascore_mdl_acc_ = None
        self.psfctscore_mdl_acc_ = None

        self.docked_ = False
        self.predcted_ = False


class BatchMoleculeData(object):
    def __init__(self, smiles_dict=None):
        # {hashid: (smiles, name)}
        self.smiles_dict = smiles_dict
        self.molecule_data = {}

        self._prepare_mol_data()
    
    def _prepare_mol_data(self):
        for (k, v) in self.smiles_dict.items():
            self.molecule_data[k] = MoleculeData(smiles=v[0], name=v[1], hashid=k)
        
        return self.molecule_data

        