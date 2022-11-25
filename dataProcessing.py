import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.general import strided_app

basicdata = pd.read_csv('./Data/BasicData.csv').sample(frac=1)
CorpusPath = './Data/Files/'
Corpusfiles = basicdata['Filename'].values
fn = len(Corpusfiles)

# Segement size and overlapp offset value
frameS = 256
offset = 64

spectrumCol = ['S'+str(i) for i in range(frameS)]
spectrumData = np.empty((0, frameS), int)
i = 0
Fcol = []

for file in tqdm(Corpusfiles):
    i+=1
    x, Fs = librosa.load(CorpusPath+file)
    segments = strided_app(x, L=frameS, S=offset)
    for l in range(64):
        k = np.random.randint(0, 10332, dtype=int)
        segment = segments[k]
        Fcol.append(file)
        row = np.abs(np.fft.fft(segment))
        spectrumData = np.vstack([spectrumData, row])

spectrumDF = pd.DataFrame(spectrumData, columns = spectrumCol)
spectrumDF['Filename'] = Fcol
# Combine basic data with features computed
masterdata = pd.merge(basicdata, spectrumDF, on='Filename')
masterdata.to_csv('./Data/MasterData.csv', index=False)
