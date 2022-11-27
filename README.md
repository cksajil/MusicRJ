
# MusicRJ

## A Machine Learning-Audio Signal Processing Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17DN_dJCyYJFQdBeyqKmcbwiIZH5ihE1z?usp=sharing)

### [To See Demo Video Click Here](https://www.youtube.com/watch?v=9X55T_ffNwg&t=224s)

**Project Details** 

I like music and would like to listen to it from FM radio channels. One good feature about FM channels is that we get to listen to random and unexpected songs which are not in our personal collection. Another important feature about FM channels is that they contain information about the locality. This included traffic updates, cultural events in the locality, festival details, etc. 

The programme schedule often goes in a pattern like music, talk, interview, and chat and so on. It would be nice if we can have a personal AI assistant (Mobile Application) which listens to such programs, records music which might be interesting to us, gather and summarise information from the Radio Jockey (RJ) talks. Since I have my background in audio signal processing, I was particularly interested in this project and wanted to give a try. 

<img src="https://github.com/cksajil/MusicRJ/blob/master/Images/BlockDGMSmall.png" width="650">

**Dataset**

The project use the dataset **[DataGTZAN music/speech collection](http://opihi.cs.uvic.ca/sound/music_speech.tar.gz)**. All the wav audio files should be extracted to the *Data/Files* folder.

**Python Version**

Python 3.9.12

**Virtual Environment**

*Installing Virtual Environment*
```console
python -m pip install --user virtualenv
```
*Creating New Virtual Environment*
```console
python -m venv envname
```
*Activating Virtual Environment*
```console
source envname/bin/activate
```
*Installing Packages*
```console
python -m pip install -r requirements.txt
```
### Model 1 (Simple DNN) Architecture

```console
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 32)                8224      
                                                                 
 dense_1 (Dense)             (None, 64)                2112      
                                                                 
 dense_2 (Dense)             (None, 128)               8320      
                                                                 
 dense_3 (Dense)             (None, 256)               33024     
                                                                 
 dense_4 (Dense)             (None, 512)               131584    
                                                                 
 dense_5 (Dense)             (None, 256)               131328    
                                                                 
 dense_6 (Dense)             (None, 128)               32896     
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_7 (Dense)             (None, 64)                8256      
                                                                 
 dense_8 (Dense)             (None, 2)                 130       
                                                                 
=================================================================
Total params: 355,874
Trainable params: 355,874
Non-trainable params: 0
_________________________________________________________________
```

**Excecute the following python scripts in the order given**

	 1. dataProcessing.py
	 2. dlModeling.py
	 3. realTimeTest.py

### Train and validation loss graph

![Loss graph](https://github.com/cksajil/MusicRJ/blob/master/Graphs/Train_Valiation_Loss.png)
