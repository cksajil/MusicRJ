
# MusicRJ

## A Machine Learning-Audio Signal Processing Project (Ongoing)

### [To See Demo Video Click Here](https://www.youtube.com/watch?v=9X55T_ffNwg&t=224s)

**Project Details** 

This is a Machine Learning-Audio Signal Processing Project where a real-time audio signal is classified into speech or music using Deep Neural Network and Convolutional Network. The long term goal is to create an AI personal assistant which listens to audio streams and summarize its content to the end user.

![Block diagram](https://i.ibb.co/5Y11jkp/Block-DGMSmall.png)

**Dataset**

The project use the dataset **[DataGTZAN music/speech collection](http://opihi.cs.uvic.ca/sound/music_speech.tar.gz)**. 

All the wav audio files should be extracted to the *Data/Files* folder.

**Python Version**
```
Python 3.9.12
```

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
*Upgrade PIP*
```console
python -m pip install --upgrade pip
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

### Model 1 Train and validation loss graph

![Loss graph](https://i.ibb.co/m5kczP3/Train-Valiation-Loss.png)
