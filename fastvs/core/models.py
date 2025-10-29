
import sklearn 
import os, sys

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import joblib
import pickle
import random


class BaseModel(object):
    def __init__(self, X=None, y=None, 
                 Xtest=None, ytest=None, 
                 out_dpath=None) -> None:
        self.X = X
        self.y = y
        self.Xtest = Xtest
        self.ytest = ytest
        self.out_dpath = out_dpath

        self.model_name = self.__class__.__name__
        self.model_output_fpath = os.path.join(self.out_dpath, f'{self.model_name}.joblib')
        self.accuracies = {}
        self.model_ = None
        os.makedirs(out_dpath, exist_ok=True)

    def _build_model(self):
        self.model_ = RandomForestRegressor(n_estimators=self.n_estimators, 
                                            random_state=42, n_jobs=self.ncpus)
    def train(self):
        self._build_model()
        self.model_.fit(self.X, self.y)

        # save model
        joblib.dump(self.model_, self.model_output_fpath, protocol=pickle.HIGHEST_PROTOCOL)
    
    def evaluate(self):
        y_pred = self.model_.predict(self.Xtest)
        mse = mean_squared_error(self.ytest, y_pred)
        r2 = r2_score(self.ytest, y_pred)
        pr, _e = stats.pearsonr(self.ytest, y_pred)
        sr, _e = stats.spearmanr(self.ytest, y_pred)

        idx = random.randint(0, self.Xtest.shape[0] - 11) 

        print(f'[INFO] MSE: {mse:.3f}, R2: {r2:.3f}, R(PCC): {pr:.3f}, R(SPC): {sr:.3f}')
        print('[INFO] ytrue ', self.ytest[idx:idx+10])
        print('[INFO] ypred ', y_pred[idx:idx+10])

        self.accuracies['mse'] = mse 
        self.accuracies['r2'] = r2
        self.accuracies['pcc'] = pr
        self.accuracies['spc'] = sr

        self.y_pred = y_pred

        return self.accuracies

    def predict(self, Xtest):  
        if self.model_ is None and os.path.exists(self.model_output_fpath):
            self.model_ = joblib.load(self.model_output_fpath)

        y_pred = self.model_.predict(Xtest)
        return y_pred


class MLPModel(BaseModel):
    def __init__(self, X=None, y=None, 
                 Xtest=None, ytest=None, 
                 out_dpath=None, **kwargs) -> None:
        super().__init__(X, y, Xtest, ytest, out_dpath)
        self.hidden_layers = [1024, 512, 256, 128, 64]
        self.activation = 'relu'
        self.solver = 'adam'
        self.alpha = 0.001
        self.ncpus = kwargs.pop('ncpus', 64)

        self.model_name = self.__class__.__name__
        self.model_output_fpath = os.path.join(self.out_dpath, f'{self.model_name}.joblib')

    def _build_model(self):
        if os.path.exists(self.model_output_fpath):
            print(f"[INFO] Reuse previous model file {self.model_output_fpath}")
            with open(self.model_output_fpath, 'rb') as f:
                self.model_ = joblib.load(f)
        else:
            print(f"[INFO] Build a {self.model_name} now ...")
            self.model_ = MLPRegressor(hidden_layer_sizes=self.hidden_layers, 
                            activation=self.activation, 
                            solver=self.solver, 
                            alpha=self.alpha, 
                            random_state=42)


class RFModel(BaseModel):

    def __init__(self, X, y, Xtest, ytest, out_dpath, **kwargs) -> None:
        super().__init__(X, y, Xtest, ytest, out_dpath)
        self.n_estimators = kwargs.pop('n_estimators', 100)
        self.ncpus = kwargs.pop('ncpus', 64)

    def _build_model(self):
        self.model_ = RandomForestRegressor(n_estimators=self.n_estimators, 
                                            random_state=42, max_depth=50,
                                            min_samples_leaf=5,
                                            n_jobs=self.ncpus)


class AdaBoostModel(BaseModel):

    def __init__(self, X, y, Xtest, ytest, out_dpath, **kwargs) -> None:
        super().__init__(X, y, Xtest, ytest, out_dpath)
        self.n_estimators = kwargs.pop('n_estimators', 100)
        self.n_models = kwargs.pop('n_estimators', 20)
        self.ncpus = kwargs.pop('ncpus', 64)

    def _build_model(self):
        self.base_model_ = RandomForestRegressor(n_estimators=self.n_estimators, 
                                                 random_state=42, 
                                                 max_depth=50,
                                                 min_samples_split=10,
                                                 min_samples_leaf=5,
                                                 n_jobs=self.ncpus)
        self.model_ = AdaBoostRegressor(base_estimator=self.base_model_, 
                                        n_estimators=self.n_models, 
                                        random_state=42)