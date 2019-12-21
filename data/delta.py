import librosa
from tqdm import tqdm
import numpy as np
import glob

def get_delta(filenames):
    print("start get delta ...")
    dim = 39
    delta_list = np.zeros((len(filenames), dim))
    for i, filename in tqdm(enumerate(filenames)):
        x, fs = librosa.load(filename,res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.4)
        #lifter > 0 -> liftering
        mfcc = librosa.feature.mfcc(x, sr=fs, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        mean = np.mean(delta, axis=1)
        mx = np.max(delta, axis=1)
        mn = np.min(delta, axis=1)
        delta = np.hstack((mean, mx, mn))
        delta_list[i] = delta
    print(f"delta size : {delta_list.shape}")
    return delta_list

if __name__ == '__main__':
    # Dataset1
    filenames = [ f"./dataset1/{i}.wav" for i in range(1, 101) ]

    # Dataset1 augument
    #filenames = [ f"./dataset1/{i}.wav" for i in range(1, 101) ] \
    #            + [ f"./dataset1/white_noise/{i}.wav" for i in range(1, 101) ] \
    #            + [ f"./dataset1/stretch/{i}.wav" for i in range(1, 101) ] \
    #            + [ f"./dataset1/shift/{i}.wav" for i in range(1, 101) ]

    # Dataset2
    #filenames = glob.glob("./ravdess-emotional-speech-audio/*/*")

    delta_list = get_delta(filenames)
    np.savez_compressed('delta-aver.npz', delta_list)







