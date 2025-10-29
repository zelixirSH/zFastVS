
import shutil
import pandas as pd
import os, sys
from fastvs.core.utils import run_command, make_temp_dpath, encode_string, load_configs
from fastvs.core.data import MoleculeData
from fastvs.core.io import *
from fastvs.core.slurm import run_tasks
from fastvs.core.fingerprint import FeaturizeMolecules
from fastvs.core.training import BindingScoreTrainer
from fastvs.core.docking import smiles_docking
    

if __name__ == "__main__":

    # molecules
    data = load_smiles_file("zinc_5w_example.smi")
    configs = load_configs('../fastvs/data/configs.json')
    train_batch = 5000
    rfpth = "1gpn_protein_atom_noHETATM.pdbqt"
    poc_c = [3.861,  67.520,  61.745]
    output = "output"

    data1, data2 = split_smiles_dict_by_num(data, n=5000)

    # test data processing
    if os.path.exists(f'{output}/test_scores.npy'):
        res_test = load_molecule_dataset(f'{output}/test_scores.npy')
    else:
        res_test = {}
    
    print("TESTING DATA Docking ")
    res_test = smiles_docking(
        data1, 
        output_dpath=f"{output}",
        receptor_fpath=rfpth, 
        pocket_center=poc_c,
        configs=configs,
        cofactor_fpath=None,
        molecules_dict=res_test
    )

    save_molecule_dataset(res_test, f'{output}/test_scores.npy')

    # test molecule features
    feat = FeaturizeMolecules(res_test, configs=configs, fp_types=['morgan', 'nyan'])
    feat_test = feat.get_fp(fp_data_prev={})

    scores = []
    feat_train = {}
    data_train, data3 = split_smiles_dict_by_num(data2, n=train_batch)
    for i in range(20):
        print(f"[INFO] =========== Train Model Round {i+1} ==============")
        # train data 
        if os.path.exists(f'{output}/train_scores.npy'):
            res_train = load_molecule_dataset(f'{output}/train_scores.npy')
        else:
            res_train = {}

        print("TRAINING DATA Docking ")
        res_train = smiles_docking(
            data_train, 
            output_dpath=f"{output}",
            receptor_fpath=rfpth, 
            pocket_center=poc_c,
            configs=configs,
            cofactor_fpath=None,
            molecules_dict=res_train
        )
        # generate features
        feat = FeaturizeMolecules(res_train, configs=configs, fp_types=['morgan', 'nyan'])
        feat_train = feat.get_fp(feat_train)
        save_molecule_dataset(res_train, f'{output}/train_scores.npy')
    
        # select another 2000 molecules
        data_train, data3 = split_smiles_dict_by_num(data3, n=train_batch)

        print("Train model now")
        trainer = BindingScoreTrainer(res_train, feat_train, res_test, 
                                      feat_test, output_dpath=f'{output}/models',
                                      method='MLP')
        trainer.train()
    
        scores.append([i, trainer.model.accuracies['mse'], trainer.model.accuracies['r2']] )

        df = pd.DataFrame(scores, columns=['idx', 'mse', 'r2'])
        df.to_csv(f'{output}/rf_model_scores.csv', header=True, index=False)

