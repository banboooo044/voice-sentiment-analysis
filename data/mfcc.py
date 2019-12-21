import librosa
from tqdm import tqdm
import numpy as np
import glob

def get_mfcc(filenames):
    print("start get_mfcc ...")
    dim = 39
    mfcc_list = np.zeros((len(filenames), dim))
    for i, filename in tqdm(enumerate(filenames)):
        x, fs = librosa.load(filename,res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.4)
        #lifter > 0 -> liftering
        mfcc = librosa.feature.mfcc(x, sr=fs, n_mfcc=13)
        mean = np.mean(mfcc, axis=1)
        mx = np.max(mfcc, axis=1)
        mn = np.min(mfcc, axis=1)
        mfcc = np.hstack((mean, mx, mn))
        mfcc_list[i] = mfcc
    print(f"mfcc size : {mfcc_list.shape}")
    return mfcc_list

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

    mfcc_list = get_mfcc(filenames)
    np.savez_compressed('mfcc-aver.npz', mfcc_list)
    
