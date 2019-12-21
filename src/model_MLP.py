import os,sys
sys.path.append('../')

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import ReLU, PReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from src.model import Model
from src.util import Util

from scipy.sparse import issparse


# tensorflowの警告抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ModelMLP(Model):
    def __init__(self, run_fold_name, **params):
        super().__init__(run_fold_name, params)

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        print(tr_y)
        # scaling
        validation = va_x is not None

        # パラメータ
        nb_classes = 8
        input_dropout = self.params['input_dropout']
        hidden_layers = int(self.params['hidden_layers'])
        hidden_units = int(self.params['hidden_units'])
        hidden_activation = self.params['hidden_activation']
        hidden_dropout = self.params['hidden_dropout']
        batch_norm = self.params['batch_norm']
        optimizer_type = self.params['optimizer']['type']
        optimizer_lr = self.params['optimizer']['lr']
        batch_size = int(self.params['batch_size'])
        nb_epoch = int(self.params['nb_epoch'])

        if issparse(tr_x):
            scaler = StandardScaler(with_mean=False)
        else:
            scaler = StandardScaler()
        scaler.fit(tr_x)
        tr_x = scaler.transform(tr_x)
        tr_y = np_utils.to_categorical(tr_y, num_classes=nb_classes)
        if validation:
            va_x =  scaler.transform(va_x)
            va_y = np_utils.to_categorical(va_y, num_classes=nb_classes)

        self.model = Sequential()
        # input layer
        self.model.add(Dropout(input_dropout, input_shape=(tr_x.shape[1],)))
        # 中間層
        for i in range(hidden_layers):
            self.model.add(Dense(hidden_units))
            if batch_norm == 'before_act':
                self.model.add(BatchNormalization())
            if hidden_activation == 'prelu':
                self.model.add(PReLU())
            elif hidden_activation == 'relu':
                self.model.add(ReLU())
            else:
                raise NotImplementedError
            self.model.add(Dropout(hidden_dropout))

        # 出力層
        self.model.add(Dense(nb_classes, activation='softmax'))

        # オプティマイザ
        if optimizer_type == 'sgd':
            optimizer = SGD(lr=optimizer_lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif optimizer_type == 'adam':
            optimizer = Adam(lr=optimizer_lr, beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError

        # 目的関数、評価指標などの設定
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # エポック数、アーリーストッピング
        # あまりepochを大きくすると、小さい学習率のときに終わらないことがあるので注意
        patience = 12
        # 学習の実行
        if validation:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                            verbose=2, restore_best_weights=True)
            history = self.model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=batch_size, verbose=2,
                                validation_data=(va_x, va_y), callbacks=[early_stopping])
        else:
            history = self.model.fit(tr_x, tr_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)

        self.scaler = scaler
    
    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        self.train(tr_x, tr_y, va_x, va_y)
        return self

    def predict(self, te_x):
        te_x = self.scaler.fit_transform(te_x)
        y_pred = self.model.predict(te_x)
        return y_pred

    def score(self, te_x, te_y):
        y_pred = self.predict(te_x)
        return accuracy_score(te_y, np.argmax(y_pred, axis=1))

    def save_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)
