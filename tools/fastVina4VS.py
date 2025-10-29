
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
    if len(sys.argv) < 2:
        print("usage: python fastAF2vs.py smiles.smi output train_batch_size")
        sys.exit(0)

    # molecules
    data = load_smiles_file(sys.argv[1])
    print("Molecules loaded!!!")
    configs = load_configs('configs.json')
    train_batch = int(sys.argv[3])
    test_size = train_batch
    rfpth = "receptor_coenzyme.pdbqt_mgltools.pdbqt"
    poc_c = [26,  30,  -8]
    output = sys.argv[2]
    method = "MLP"
    data1, data2 = split_smiles_dict_by_num(data, n=test_size)

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
    feat_test_fpath = f'{output}/test_features.npy'
    if os.path.exists(feat_test_fpath):
        feat_test = load_molecule_dataset(feat_test_fpath)
        print(list(feat_test.items())[:5])
    else:
        feat = FeaturizeMolecules(res_test, configs=configs, fp_types=['morgan', 'nyan'])
        feat_test = feat.get_fp(fp_data_prev={})
        #save_molecule_dataset(feat_test, feat_test_fpath)

    scores = []
    # train molecule features
    #feat_train_fpath = f'{output}/train_features.npy'
    if os.path.exists(feat_test_fpath):
        #print(f'[INFO] find previous feature file {feat_train_fpath}')
        feat_train = load_molecule_dataset(feat_test_fpath)
    else:
        feat_train = {}

    data_train, data3 = split_smiles_dict_by_num(data2, n=train_batch)
    for i in range(1000):
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
        feat_train = feat.get_fp(fp_data_prev=feat_train)
        #save_molecule_dataset(feat_train, feat_train_fpath)
        save_molecule_dataset(res_train, f'{output}/train_scores.npy')

        print("Train model now")
        trainer = BindingScoreTrainer(res_train, feat_train, res_test, 
                                      feat_test, output_dpath=f'{output}/models', method=method)
        trainer.train()

        # record predictions
        test_pred_true = trainer.test_pred_true_
        os.makedirs(f'{output}/ypred', exist_ok=True)
        test_pred_true.to_csv(f'{output}/ypred/test_pred_true_iter{i}.csv', header=True, index=False)
    
        scores.append([i, trainer.model.accuracies['mse'], trainer.model.accuracies['r2'], 
                       trainer.model.accuracies['spc'], trainer.model.accuracies['pcc']] )
        # save model performance
        df = pd.DataFrame(scores, columns=['idx', 'mse', 'r2', 'spc', 'pcc'])
        df.to_csv(f'{output}/{method}_model_scores.csv', header=True, index=False)

        # select another n molecules
        data_train, data3 = split_smiles_dict_by_num(data3, n=train_batch)

