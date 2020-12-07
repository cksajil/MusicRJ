# Import Libraries
import librosa
import numpy as np
import pandas as pd
from sklearn import preprocessing


basicdata = pd.read_csv('./Data/MscReviewed/metadata.tsv',
            usecols =['speechpath', 'speaker_gender'], 
            sep='\t')


basicdata = basicdata[basicdata['speaker_gender'].isin(['Male','Female'])]
basicdata = basicdata.replace({'Male': 1, 'Female': 0})

CorpusPath = './Data/MscReviewed/'
Corpusfiles = basicdata['speechpath'].values
fn =len(Corpusfiles)


def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))


frameS = 256
offset = 64


spectrumCol = ['S'+str(i) for i in range(frameS)]
spectrumData = np.empty((0, frameS), int)

i=0
for file in Corpusfiles:
    print(str(fn-i)+ ' to go', end='\r', flush=True)
    i+=1
    x, Fs = librosa.load(CorpusPath+file)
    segments = strided_app(x, L = frameS, S = offset)
    freqData = np.empty((0, frameS), int)
    for segment in segments:
        row = np.abs(np.fft.fft(segment))
        freqData = np.vstack([freqData,row])
    spectrumData = np.vstack([spectrumData, np.mean(freqData, axis=0)])


min_max_scaler = preprocessing.MinMaxScaler()
spectrumDataScaled = min_max_scaler.fit_transform(spectrumData)
spectrumDF = pd.DataFrame(spectrumDataScaled, columns = spectrumCol)
spectrumDF['speechpath'] = Corpusfiles


# Combine All Features
masterdata = pd.merge(basicdata, spectrumDF, on='speechpath')
masterdata.to_csv('Data/MasterData.csv', index=False)