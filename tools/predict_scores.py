

import os, sys 
from fastvs.core.models import * 
from fastvs.core.utils import run_command, make_temp_dpath, encode_string, load_configs
from fastvs.core.data import BatchMoleculeData
from fastvs.core.io import *
from fastvs.core.training import BindingScoreTrainer
from fastvs.core.slurm import run_tasks
from fastvs.core.fingerprint import FeaturizeMolecules
from fastvs.core.docking import smiles_docking
import joblib
import numpy as np
import random
import pandas as pd


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: predict_scores.py input.smiles output/ MLPModel.joblib")
        sys.exit(0)

    inp = sys.argv[1]
    output = sys.argv[2]
    mdl = sys.argv[3]

    batch_size= 100000
    model_method = "MLP"
    rfpth = "receptor_coenzyme.pdbqt_mgltools.pdbqt"
    poc_c = [26,  30,  -8]

    configs = load_configs('configs.json')
    data = load_smiles_file(inp)
    data, data_remain = split_smiles_dict_by_num(data, n=1000, seed=-1)

    print("TESTING DATA Docking ")
    mol_data = smiles_docking(
        data, 
        output_dpath=f"{output}",
        receptor_fpath=rfpth, 
        pocket_center=poc_c,
        configs=configs,
        cofactor_fpath=None,
        molecules_dict={},
        remove_temp=False,
    )

    model = joblib.load(mdl)

    results = {}

    # features
    feat = FeaturizeMolecules(mol_data, configs=configs, 
                              fp_types=['morgan', 'nyan'])
    feat_test = feat.get_fp(fp_data_prev={})

    # predict data
    trainer = BindingScoreTrainer(mol_data, feat_test, None, 
                                  None, output_dpath=f'{output}/models', 
                                  method='MLP')
    Xtest = trainer.X 
    ytest = trainer.y
    ypred = model.predict(Xtest)
    
    for (i, k) in enumerate(list(feat_test.keys())):
        results[k] = [mol_data[k].smiles, ypred[i]]
        if i < 20:
            print(k, mol_data[k].smiles, mol_data[k].vinascore, ypred[i])
    
    # model accuracy
    mse = mean_squared_error(ytest, ypred)
    r2 = r2_score(ytest, ypred)

    idx = random.randint(0, Xtest.shape[0] - 11) 
    print('[INFO] MSE: %.3f, R2: %.3f' % (mse, r2))
    print('[INFO] ytrue ', ytest[idx:idx+10])
    print('[INFO] ypred ', ypred[idx:idx+10])

    rnd = 1
    results_df = pd.DataFrame()
    while len(data_remain) > 0:
        print(f"[INFO] ======= Processing data round {rnd} =======")
        # prepare molecule data
        data, data_remain = split_smiles_dict_by_num(data_remain, n=batch_size)
        mol_data = BatchMoleculeData(data).molecule_data

        # features
        feat = FeaturizeMolecules(mol_data, configs=configs, fp_types=['morgan', 'nyan'])
        feat_test = feat.get_fp(fp_data_prev={})

        # predict data
        trainer = BindingScoreTrainer(mol_data, feat_test, None, 
                                      None, output_dpath=f'{output}/models', 
                                      method='MLP')
        Xtest = trainer.X 
        ytest = trainer.y
        ypred = model.predict(Xtest)

        for (i, k) in enumerate(list(feat_test.keys())):
            results[k] = [mol_data[k].smiles, ypred[i]]

        dat = pd.DataFrame()
        dat['hashid'] = list(results.keys())
        dat['smiles'] = [x[0] for x in list(results.values())]
        dat['pred_score'] = [x[1] for x in list(results.values())]
        
        '''if rnd == 1:
            results_df = dat
        else:
            results_df = pd.concat([results_df, dat], axis=0)'''
            
        dat.to_csv(f'{output}/predicted_scores.csv', 
                          header=True, index=True, sep=" ", 
                          float_format="%.3f")
        
        rnd += 1

