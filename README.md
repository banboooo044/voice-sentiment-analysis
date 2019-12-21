# 音声処理からの感情解析

## 1. データの準備
* Dataset1 : dataset1.zipを解凍
* Dataset2 : The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)

1. https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio
[Download] からダウンロード

2. zipファイルを解凍し, 内部のravdess-emotional-speech-audio/audio_speech_actors_01-24 ディレクトリを削除

3. data/ravdess-emotional-speech-audio に置く.

## 2. セットアップ方法

1. 以下をコマンドラインで実行.
   ```
   git clone https://github.com/banboooo044/voice-sentiment-analysis.git
   ```
2. 1の方法でデータの準備をする
3. 以下のようにプログラムを実行.特徴量の行列ファイルが作成される.
   ```bash
    bash data/setup.sh
   ```

## 3. 特徴量作成

### data/
* mfcc.py : メルケプストラムから得たスペクトル包絡の行列データを作成
* delta.py : デルタメルケプストラムの行列データを作成
* power.py : パワーの行列データを作成

## 4. 分析プログラム
### src/ : 汎用的なプログラム(他のプログラムから呼び出して使う再利用性の高いもの)

* model.py : Modelクラスは学習, 予測, モデルの保存やロードを行う.Modelクラスを継承して, 分類アルゴリズムごとのクラスを作る.
  * model_SVC : ナイーブベイズ(多項分布モデル)
  * model_MLP : Multilayer Perceptron
  
* runner.py : Runnerクラスはクロスバリデーションなども含めた学習, 評価, 予測を行うためのクラス.Modelクラスを継承しているプログラムを渡す.
(読み込むデータの種類の変更 Dataset1 or Dataset1-augmented or Dataset2 は手動でRunner.load_x_train() のところを直して)

* util.py : ファイルの入出力, ログの出力や表示, 計算結果の表示や出力を行うクラス

### code-analysis/ : コード分析用のプログラム

* PCA_plot.py : PCAの結果の図作成のプログラム
* permutation_importance.py : 特徴量の重要度の考察で用いたプログラム
* run_[アルゴリズム名].py : Runnerクラスを用いて, 実際に各分類アルゴリズムで学習を行うプログラム.
* [アルゴリズム名]_gridCV.py : グリッドサーチでアルゴリズムのパラメータチューニングを行うプログラム.
* [アルゴリズム名]_tuning.py : hyperopt(ベイズ最適化を用いたパラメータ自動探索ツール)を用いてパラメータチューニングを行うプログラム

