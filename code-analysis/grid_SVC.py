import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from src.model_SVC import ModelSVC
from scipy import sparse

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from src.util import Logger
from src.runner import Runner

logger = Logger()

if __name__ == '__main__':
    params = {
        'kernel' : 'linear',
        'gamma' : 0.001
    }
    params_SVC = dict(params)
    
    param_grid_ = [
        {'n_components' : [ 10, 30, 50, 100], 'n_iter' : [8, 16], 'C': [1, 10, 100, 1000] },
        { 'apply_svd': [False], 'C': [1, 10, 100, 1000] }
    ]
    
    feature = [
        ["mfcc", "delta", "power"]
    ]

    results = [ ]
    x = Runner.load_x_train(feature)
    y = Runner.load_y_train()
    model = ModelSVC("SVC", **params_SVC)
    search = GridSearchCV( model, cv=5, param_grid=param_grid_ , return_train_score=True )
    search.fit(x, y)
    results.append( (search, feature) )
    logger.info(f'{feature} - bestscore : {search.best_score_} - result :{search.best_params_}')


    for search, name in results:
        logger.info(f'{name} - bestscore : {search.best_score_}')
    