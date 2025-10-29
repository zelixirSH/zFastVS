
import sklearn 
import os, sys
from fastvs.core.models import *
import pandas as pd


class BindingScoreTrainer(object):

    def __init__(self, molecules_dict_train=None, features_train=None, 
                 molecules_dict_test=None, features_test=None, 
                 output_dpath=None, 
                 method='Adaboost') -> None:
        
        if features_train is not None:
            self.X, self.y = self._prepare_data(molecules_dict_train, features_train)
        if features_test is not None:
            self.Xtest, self.ytest = self._prepare_data(molecules_dict_test, features_test)
        self.method = method 
        self.output_dpath = output_dpath

    def _prepare_data(self, molecules_dict, features):

        # prepare data
        X = list(features.values())

        try:
            y = [molecules_dict[x].vinascore for x in list(features.keys())]
            y = [x if x < 9.9 else 9.9 for x in y]
        except:
            y = [9.9, ] * len(X) 
            
        return np.array(X), np.array(y)
    
    def train(self):
        print(f"[INFO] Training data shape X={self.X.shape} and y={self.y.shape}")
        print(f"[INFO] Test data shape X={self.Xtest.shape} and y={self.ytest.shape}")
        
        # train model
        if self.method == 'RF':
            self.model = RFModel(self.X, self.y, 
                                 self.Xtest, self.ytest, 
                                 self.output_dpath, 
                                 n_estimators=200)
        elif self.method == 'Adaboost':
            self.model = AdaBoostModel(self.X, self.y, 
                                       self.Xtest, self.ytest, 
                                       self.output_dpath, 
                                       n_estimators=100, 
                                       n_models=50,)
        elif self.method == 'MLP':
            self.model = MLPModel(self.X, self.y, 
                                self.Xtest, self.ytest, 
                                self.output_dpath, 
                                ncpus=32,)
        
        #print(f"[INFO] Training model with {self.method}")
        self.model.train()

        # evaluate model
        print("[INFO] Model evaluate")
        self.model.evaluate()

        # test data
        self.test_pred_true_ = pd.DataFrame()
        self.test_pred_true_['ypred'] = self.model.y_pred
        self.test_pred_true_['ytrue'] = self.model.ytest

