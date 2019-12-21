import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from src.model import Model
from src.util import Util

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD

from scipy.sparse import issparse

np.random.seed(71)
class ModelSVC(Model, BaseEstimator, ClassifierMixin):
    def __init__(self, run_fold_name, C=1.0, kernel='rbf', degree=3, gamma='scale', 
                        coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
                        class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', apply_svd = True,
                        random_state=None,svd_iter = 5, n_components=100):
        
        params_SVC = {
            'C' : C,
            'kernel' : kernel,
            'degree' : degree,
            'gamma' : gamma,
            'coef0' : coef0,
            'shrinking' : shrinking,
            'probability' : probability,
            'tol' : tol,
            'cache_size' : cache_size, 
            'class_weight' : class_weight,
            'verbose'  : verbose,
            'max_iter' : max_iter, 
            'decision_function_shape' : decision_function_shape, 
            'random_state'  : random_state
        }
        params_svd = {
            'random_state' : random_state,
            'n_iter' : svd_iter,
            'n_components' : n_components
        }
        super().__init__(run_fold_name, {**params_SVC, **params_svd})
        self.model = SVC(**params_SVC)
        self.svd = TruncatedSVD(**params_svd)
        self.apply_svd = apply_svd
        
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # データのセット
        if issparse(tr_x):
            scaler1 = StandardScaler(with_mean=False)
        else:
            scaler1 = StandardScaler()

        scaler_sc1 = scaler1.fit(tr_x)
        tr_x = scaler_sc1.transform(tr_x)
        self.scaler1 = scaler1

        if self.apply_svd:
            self.svd = self.svd.fit(tr_x)
            tr_x = self.svd.transform(tr_x)

            if issparse(tr_x):
                scaler2 = StandardScaler(with_mean=False)
            else:
                scaler2 = StandardScaler()

            scaler_sc2 = scaler2.fit(tr_x)
            tr_x = scaler_sc2.transform(tr_x)
            self.scaler2 = scaler2

        self.model = self.model.fit(tr_x, tr_y)
        
    def fit(self, tr_x, tr_y):
        self.train(tr_x, tr_y)
        return self

    def predict(self, te_x):
        te_x = self.scaler1.transform(te_x)
        if self.apply_svd:
            te_x = self.svd.transform(te_x)
            te_x = self.scaler2.transform(te_x)
        return self.model.predict(te_x)

    def score(self, te_x, te_y):
        y_pred = self.predict(te_x)
        return accuracy_score(te_y, y_pred)

    def get_params(self, deep=True):
        dic = self.model.get_params(deep)
        dic["run_fold_name"] = self.run_fold_name 
        return dic
    
    def set_params(self, **parameters):
        self.run_fold_name = parameters.get("run_fold_name", "")
        parameters.pop("run_fold_name", None)
        self.params.update(parameters)
        params_svd = {}
        if 'random_state' in parameters:
            params_svd['random_state'] = parameters['random_state']
        if 'n_iter' in parameters:
            params_svd['n_iter'] = parameters['n_iter']
            parameters.pop('n_iter')
        if 'n_components' in parameters:
            params_svd['n_components'] = parameters['n_components']
            parameters.pop('n_components')
        if 'apply_svd' in parameters:
            self.apply_svd = parameters['apply_svd']
            parameters.pop('apply_svd')
        
        self.svd.set_params(**params_svd)
        self.model.set_params(**parameters)
        return self
    
    def save_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)
