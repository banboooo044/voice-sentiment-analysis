import os,sys
sys.path.append('../')

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import LeaveOneGroupOut
from typing import Callable, List, Optional, Tuple, Union
from sklearn.model_selection import learning_curve
from scipy import sparse
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from src.model import Model
from src.util import Logger, Util

sns.set()
logger = Logger()

class RunnerLeaveOneOut:
    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model], features: str, params: dict):
        """コンストラクタ
        :param run_name: ランの名前
        :param model_cls: モデルのクラス
        :param features: 特徴量のリスト
        :param params: ハイパーパラメータ
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.params = params
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.n_fold = 100

    def train_fold(self, i_fold: Union[int, str]) -> Tuple[
        Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """クロスバリデーションでのfoldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる
        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        validation = i_fold != 'all'
        train_x = RunnerLeaveOneOut.load_x_train(self.features)
        train_y = RunnerLeaveOneOut.load_y_train()
        if validation:
            # 学習データ・バリデーションデータをセットする
            tr_idx, va_idx = self.load_index_fold(i_fold)
            tr_x, tr_y = train_x[tr_idx], train_y[tr_idx]
            va_idx = np.array([np.min(va_idx)])
            va_x, va_y = train_x[va_idx], train_y[va_idx]
            
            # 学習を行う
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)

            # バリデーションデータへの予測・評価を行う
            va_pred = model.predict(va_x)
            score = model.score(va_x, va_y)

            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, score
        else:
            # 学習データ全てで学習を行う
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            # モデルを返す
            return model, None, None, None

    def run_train_cv(self, lr_curve_train_sizes: Optional[List[int]]=None) -> None:
        """クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []
        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model(self.features)

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 予測結果の保存
        Util.dump(preds, f'../model/pred/{self.features}/{self.run_name}-train.pkl')

        # 評価結果の保存
        logger.result_scores(self.run_name, scores)
            


    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction cv')

        test_x = RunnerLeaveOneOut.load_x_test(self.features)

        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model(self.features)
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Util.dump(pred_avg, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')

    def run_train_all(self) -> None:
        """学習データすべてで学習し、そのモデルを保存する"""
        logger.info(f'{self.run_name} - start training all')

        # 学習データ全てで学習を行う
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model(self.features)

        logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self) -> None:
        """学習データすべてで学習したモデルにより、テストデータの予測を行う
        あらかじめrun_train_allを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction all')

        test_x = RunnerLeaveOneOut.load_x_test(self.features)

        # 学習データ全てで学習したモデルで予測を行う
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model(self.features)
        pred = model.predict(test_x)

        # 予測結果の保存
        Util.dump(pred, f'../model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction all')

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, **self.params)

    @classmethod
    def load_x_train(cls, features) -> np.array:
        """学習データの特徴量を読み込む
        :return: 学習データの特徴量
        """
        x = None
        if "mfcc" in features:
            matrix = np.load('../data/mfcc-dataset1-aver.npz')['arr_0']
            #matrix = np.load('../data/mfcc-dataset1-a-aver.npz')['arr_0']
            if x is None:
                x = matrix
            else:
                x = np.hstack(( x, matrix))
        if "delta" in features:
            matrix = np.load('../data/delta-dataset1-aver.npz')['arr_0']
            #matrix = np.load('../data/delta-dataset1-a-aver.npz')['arr_0']
            if x is None:
                x = matrix
            else:
                x = np.hstack(( x, matrix))
        if "power" in features:
            matrix = np.load('../data/power-dataset1-aver.npz')['arr_0']
            #matrix = np.load('../data/power-dataset1-a-aver.npz')['arr_0']
            if x is None:
                x = matrix
            else:
                x = np.hstack(( x, matrix))
        return x

    @classmethod
    def load_y_train(cls) -> np.array:
        """学習データの目的変数を読み込む
        :return: 学習データの目的変数
        """
        # Dataset1
        labels = np.load('../data/label-dataset1.npz')['arr_0'].astype('int')

        # Dataset1 Augumented
        # label = np.load('../data/label-dataset1.npz')['arr_0'].astype('int')
        # labels = np.array([ deepcopy(label) for _ in range(4) ]).flatten()
        
        return labels

    @classmethod
    def load_x_test(cls, features) -> np.array:
        """テストデータの特徴量を読み込む
        :return: テストデータの特徴量
        """
        return np.load('../vec/hogehoge.npy', allow_pickle=True)
    

    def load_index_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す
        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        train_y = RunnerLeaveOneOut.load_y_train()
        dummy_x = np.zeros(len(train_y))
        skf = LeaveOneGroupOut()
        return list(skf.split(dummy_x, train_y, groups=[i%100 for i in range(len(train_y))]))[i_fold]