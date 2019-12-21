import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from src.runner import Runner
from src.model_SVC import ModelSVC

if __name__ == '__main__':
    params = {
        'C' : 100,
        'n_components' : 15,
        'svd_iter' : 8,
        'kernel' : 'linear',
        'random_state': 71,
        'apply_svd' : True
    }

    features = [
        "mfcc", "delta", "power"
    ]

    params_SVC = dict(params)
    runner = Runner(run_name='svc', model_cls=ModelSVC, features=features, params=params_SVC)

    # 1回実行
    # runner.train_fold(0)
    # クロスバリデーションで実行
    runner.run_train_cv()
