
import shutil
import os, sys
from fastvs.core.utils import run_command, make_temp_dpath, encode_string, load_configs
from fastvs.core.data import MoleculeData
from fastvs.core.io import smiles2pdb, split_smiles_dict, load_smiles_file, save_molecule_dataset, load_molecule_dataset
from fastvs.core.slurm import run_tasks


PACKAGE_DPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")


class MolecularDocking(object):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.receptor_fpath = kwargs['receptor']
        self.pocket_center  = kwargs['pocket_center']
        self.pocket_size    = kwargs['pocket_size']
        self.ligand_fpath   = kwargs['ligand']
        self.cofactor_fpath = kwargs['cofactor']
        self.output_dpath   = kwargs['output_dpath']
        self.configs        = kwargs['configs'] #[0]

        self.pocket_fpath   = None
        self.scorer_name = self.__class__.__name__ 

        #print("pacdock configs", self.configs)

        self.docked_fpath= os.path.join(self.output_dpath, "docked_ligand.pdbqt")
        self.sfct_fpath  = os.path.join(self.output_dpath, "sfct_scores.txt")
        self.vinascores  = None 
        self.sfctscores  = None
    
    def _prepare_xyz(self) -> str:
        """Write the pocket center into a parameter file pocket.xyz. 

        Returns:
            str: pocket.xyz file path
        """
        os.makedirs(self.output_dpath, exist_ok=True)
        #print("Copy Ligand File", self.ligand_fpath, os.path.join(self.output_dpath, "ligand.pdb"))
        #shutil.copy(self.ligand_fpath, os.path.join(self.output_dpath, "ligand.pdb"))

        with open(os.path.join(self.output_dpath, "pocket.xyz"), 'w') as tofile:
            tofile.write(f"{self.pocket_center[0]} {self.pocket_center[1]} {self.pocket_center[2]}\n")

        self.pocket_fpath = os.path.join(self.output_dpath, "pocket.xyz")
        self.ligand_fpath = os.path.join(self.output_dpath, "ligand.pdb")
        return self.pocket_fpath

    def _get_docking_cmd(self):
        """Run molecule docking with Vina or SFCT.
        """

        if self.pocket_fpath is None or (not os.path.exists(self.pocket_fpath)):
            self._prepare_xyz()

        if self.cofactor_fpath is not None and os.path.exists(self.cofactor_fpath):
            cmd = f"{self.configs['sfct']['script']} {self.receptor_fpath} {self.ligand_fpath} " + \
                f"{self.pocket_fpath} {self.output_dpath} {self.pocket_size} {self.cofactor_fpath}"
        else:
            cmd = f"{self.configs['sfct']['script']} {self.receptor_fpath} {self.ligand_fpath} " + \
                f"{self.pocket_fpath} {self.output_dpath} {self.pocket_size}"

        if not os.path.exists(self.docked_fpath):
            return cmd
        else:
            return ""
    
    def _parse_output(self):
        """Parse the output file and get the docking scores.
        """
        if self.mode == "sfct":
            try:
                with open(self.sfct_fpath) as lines:
                    lines = [x for x in lines if "#" not in x]
                    self.vinascores = [float(x.split()[2]) for x in lines]
                    self.sfctscores = [float(x.split()[3]) for x in lines]
            except:
                self.vinascores = [9.9, ] * 20
                self.sfctscores = [5.0, ] * 20
        else:
            with open(self.docked_fpath) as lines:
                self.vinascores = [float(x.split()[3]) for x in lines 
                                   if "REMARK VINA RESULT:" in x]

    def run(self):
        # run docking
        cmd = self._get_docking_cmd()
        run_command(cmd)

        # parse results 
        self._parse_output()

        return self


class VinascoreScorer(object):
    """One of the enzyme design scorer. 
    VinascoreScorer defines the docking scores of the docking poses.

    Method
    ------
    score: given a protein-ligand docking poses file, return docking score.

    Args:
        BaseScorer (object): base score.
        pose_fpath: str, docking pose file, pdbqt format
        pose_type: str, docking pose file type, generally vina, 
                   idock or gnina, ledock
    """

    def __init__(self, **kwargs):

        super().__init__()
        
        #self.modeller     = kwargs['modeller']
        self.pose_fpath   = kwargs['pose_fpath']
        #self.output_dpath = kwargs['output_dpath']
        #self.configs      = kwargs['configs']
        self.pose_scores_type = kwargs['pose_type']
        #self.scorer_name  = self.__class__.__name__
    
    def _parse_vina_result(self) -> list:
        """Parse Vina PDBQT file for docking scores.

        Returns:
            scores: list, scores of docking poses.
        """

        if not os.path.exists(self.pose_fpath):
            return [0.0, ] * 10

        if self.pose_scores_type == "general":
            return [0.0, ] * 10
        elif self.pose_scores_type in ["idock", "ledock", 'gnina', 'tdock', 
                                       'vina', 'gnina_energy', 'gnina_cnn', 'sfct']:
            scores = None
            with open(self.pose_fpath) as lines:
                lines = [x for x in lines]
                if len(lines) == 0:
                    return [0.0, ] * 10

                if self.pose_scores_type == "idock":
                    scores = [float(x.split()[-2]) for x in lines
                            if "REMARK 922        TOTAL FREE ENERGY PREDICTED BY IDOCK" in x]
                elif self.pose_scores_type == "ledock":
                    scores = [float(x.split()[-2]) for x in lines if "REMARK Cluster" in x]
                elif self.pose_scores_type == "gnina_cnn":
                    scores = [-1. * float(x.split()[-4][:-6]) for x in lines if "REMARK minimizedAffinity" in x]
                    # names = [x.split()[-1].strip("\n") for x in lines if "REMARK  Name =" in x]
                elif self.pose_scores_type == "gnina_energy":
                    scores = [-1. * float(x.split()[2][:-6]) for x in lines if "REMARK minimizedAffinity" in x]
                    # names = [x.split()[-1].strip("\n") for x in lines if "REMARK  Name =" in x]
                elif self.pose_scores_type == "tdock":
                    scores = [float(x.split()[-2]) for x in lines if "REMARK VINA ENERGY" in x]
                elif self.pose_scores_type.lower() == "vina":
                    scores = [float(x.split()[-3]) for x in lines if "REMARK VINA RESULT:" in x]
                elif self.pose_scores_type.lower() == "sfct":
                    scores = [float(x.split()[2]) for x in lines if "#" not in x]
                else:
                    scores = [0.0, ] * 10

            return scores
        else:
            return [0.0, ] * 10
    
    def score(self):

        return self._parse_vina_result()


def smiles_docking(smiles_dict, 
                   output_dpath=None,
                   receptor_fpath=None, 
                   pocket_center=None, 
                   pocket_size=15, 
                   cofactor_fpath=None,
                   configs=None, 
                   molecules_dict={},
                   remove_temp=True,
                   ):
    cmd_list = []
    out_dict = {}

    # prepare data for docking now
    for hashid in list(smiles_dict.keys()):

        smiles = smiles_dict[hashid][0]
        name   = smiles_dict[hashid][1]

        # create molecule object
        #if molecule is None:
        if hashid not in molecules_dict.keys():
            molecule = MoleculeData(smiles, name, hashid)
            molecules_dict[hashid] = molecule
        else:
            molecule = molecules_dict[hashid]

        if molecule.docked_:
            continue 

        # make temp input_ligand file
        task_out_dpath = os.path.join(output_dpath, "docking", hashid[:2], hashid)
        os.makedirs(task_out_dpath, exist_ok=True)
        ligand_fpath = os.path.join(task_out_dpath, "ligand.pdb")
        
        ob_cmd = smiles2pdb(smiles, out_dpath=task_out_dpath, return_cmd=True)
        #print("OBABEL CMD ", ob_cmd)

        # define docking results
        vina_fpath = os.path.join(task_out_dpath, "docked_ligand.pdbqt")
        sfct_fpath = os.path.join(task_out_dpath, "sfct_results.dat")
        out_dict[hashid] = [vina_fpath, sfct_fpath]

        if os.path.exists(vina_fpath):
            continue

        # run docking
        dock = MolecularDocking(
                    receptor=receptor_fpath,
                    ligand=ligand_fpath,
                    pocket_center=pocket_center,
                    pocket_size=pocket_size,
                    cofactor=cofactor_fpath,
                    output_dpath=task_out_dpath,
                    configs=configs,
                )
        cmd = dock._get_docking_cmd()
        if len(cmd):
            if len(ob_cmd):
                cmd_list.append(ob_cmd + " && " + cmd)
            else:
                cmd_list.append(cmd)
    
    # run docking now
    print("Running cmd ", cmd_list[:2])
    if len(cmd_list):
        bs = int(len(cmd_list) / 200)
        if bs >= 50:
            bs = 50
        if bs < 10:
            bs = 10
        
        run_tasks(cmd_list, batch_size=bs, queue='other', ncpus=16, max_tasks=40)
        #run_tasks(cmd_list, batch_size=50)
        # run task locally
        '''for cmd in cmd_list:
            run_command(cmd)'''
            
    # collect data
    for hashid in list(smiles_dict.keys()):
        molecule = molecules_dict[hashid]
        task_out_dpath = os.path.join(output_dpath, "docking", hashid[:2], hashid)

        if molecule.docked_:
            continue

        if os.path.exists(out_dict[hashid][0]):
            #sfct = VinascoreScorer(pose_fpath=out_dict[hashid][1], pose_type='sfct')
            #molecule.sfctscore = sorted(sfct.score())

            vina = VinascoreScorer(pose_fpath=out_dict[hashid][0], pose_type='vina')
            molecule.vinascore = list(sorted(vina.score()))[0]
        
            molecule.docked_ = True
            molecules_dict[hashid] = molecule
        else:
            print("[WARNING] no docking output for ", hashid)
            #molecules_dict.pop(hashid)
            molecule.docked_ = True
            molecule.vinascore = 9.9
            molecules_dict[hashid] = molecule
    
    if remove_temp:
        cmd = "cd {} && bash {}/bin/remove_multi_files.sh".format(os.path.join(output_dpath, "docking"), PACKAGE_DPATH)
        run_command(cmd, verbose=True)

    return molecules_dict


if __name__ == "__main__":
    from fastvs.core.fingerprint import FeaturizeMolecules
    from fastvs.core.training import BindingScoreTrainer
    # molecules
    data = load_smiles_file("zinc_5w_example.smi")
    configs = load_configs('../fastvs/data/configs.json')

    data1, data2 = split_smiles_dict(data, 0.1)
    data3, data4 = split_smiles_dict(data2, 0.5)
    
    # run docking now
    # train data 
    if os.path.exists('output/train_scores.npy'):
        res_train = load_molecule_dataset('output/train_scores.npy')
    else:
        res_train = {}

    print("TRAINING DATA Docking ")
    res_train = smiles_docking(
        data3, 
        output_dpath="output",
        receptor_fpath="1gpn_protein_atom_noHETATM.pdbqt", 
        pocket_center=[3.861,  67.520,  61.745],
        configs=configs,
        cofactor_fpath=None,
        molecules_dict=res_train
    )
    feat = FeaturizeMolecules(res_train, configs=configs, fp_types=['morgan', 'nyan'])
    feat_train = feat.get_fp()
    save_molecule_dataset(res_train, 'output/train_scores.npy')
    
    # test data 
    if os.path.exists('output/test_scores.npy'):
        res_test = load_molecule_dataset('output/test_scores.npy')
    else:
        res_test = {}

    print("TESTING DATA Docking ")
    res_test = smiles_docking(
        data1, 
        output_dpath="output",
        receptor_fpath="1gpn_protein_atom_noHETATM.pdbqt", 
        pocket_center=[3.861,  67.520,  61.745],
        configs=configs,
        cofactor_fpath=None,
        molecules_dict=res_test
    )

    save_molecule_dataset(res_test, 'output/test_scores.npy')
    feat = FeaturizeMolecules(res_test, configs=configs, fp_types=['morgan', 'nyan'])
    feat_test = feat.get_fp()

    print("Train model now")
    trainer = BindingScoreTrainer(res_train, feat_train, res_test, 
                                  feat_test, output_dpath='output/models')
    trainer.train()
    