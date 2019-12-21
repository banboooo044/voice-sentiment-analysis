# 基本となるパラメータ

import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from src.runner import Runner
from src.runner2 import RunnerLeaveOneOut
from src.model_MLP import ModelMLP

if __name__ == '__main__':
    base_params=  {
        'nb_class' : 4,
        'input_dropout': 0.0,
        'hidden_layers': 3,
        'hidden_units': 96,
        'hidden_activation': 'relu',
        'hidden_dropout': 0.2,
        'batch_norm': 'before_act',
        'optimizer': {'type': 'adam', 'lr': 0.001},
        'batch_size': 64,
        'nb_epoch': 1000,
    }

    params = {'batch_norm': 'no', 'batch_size': 64.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.45, 'hidden_layers': 3.0, 'hidden_units': 224.0, 'input_dropout': 0.2, 
        'optimizer': {'lr': 0.00091, 'type': 'adam'}}
    base_params.update(params)

    features = [
        "mfcc", "delta", "power"
    ]

    params_MLP = dict(base_params)

    # Dataset1
    runner = RunnerLeaveOneOut(run_name='MLP1', model_cls=ModelMLP, features=features, params=params_MLP)

    # Dataset2
    #runner = Runner(run_name='MLP1', model_cls=ModelMLP, features=features, params=params_MLP)

    runner.run_train_cv()

