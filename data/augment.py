# voice_analysis/data で実行すること.
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm

# 参考 : https://qiita.com/cvusk/items/aa628e84e72cdf0a6e77

# 白色ノイズを加える
def add_white_noise(x, rate=0.005):
    # ノイズ : rate*np.random.randn(len(x))
    return x + rate*np.random.randn(len(x))

# 音を伸ばす
def stretch_sound(x, rate=1.1):
    #
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x) > input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")

# 時間をずらす
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

# wav file の保存
def write_wav(filename, x, fs):
    librosa.output.write_wav(filename, x, fs)

def augument(filenames):
    # ディレクトリの作成
    os.makedirs('./dataset1/white_noise', exist_ok=True)
    os.makedirs('./dataset1/stretch', exist_ok=True)
    os.makedirs('./dataset1/shift', exist_ok=True)
    
    for idx, filename in tqdm(enumerate(filenames)):
        x, fs = librosa.load(filename,res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.4)

        white = add_white_noise(x)
        stretch = stretch_sound(x)
        shift = shift_sound(x)

        write_wav(f"./dataset1/white_noise/{idx+1}.wav", white, fs)
        write_wav(f"./dataset1/stretch/{idx+1}.wav", stretch, fs)
        write_wav(f"./dataset1/shift/{idx+1}.wav", shift, fs)

if __name__ == "__main__":
    # Dataset1
    filenames = [ f"./dataset1/{i}.wav" for i in range(1, 101) ]
    augument(filenames)