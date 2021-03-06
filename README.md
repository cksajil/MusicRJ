
# MusicRJ

## A Machine Learning-Audio Signal Processing Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17DN_dJCyYJFQdBeyqKmcbwiIZH5ihE1z?usp=sharing)


**Project Details**: I like music and would like to listen to it from FM radio channels. One good feature about FM channels is that we get to listen to random and unexpected songs which are not in our personal collection. Another important feature about FM channels is that they contain information about the locality. This included traffic updates, cultural events in the locality, festival details, etc. 

The programme schedule often goes in a pattern like music, talk, interview, and chat and so on. It would be nice if we can have a personal AI assistant (Mobile Application) which listens to such programs, records music which might be interesting to us, gather and summarise information from the Radio Jockey (RJ) talks. Since I have my background in audio signal processing, I was particularly interested in this project and wanted to give a try. 

![Block Diagram](https://github.com/cksajil/MusicRJ/blob/master/Images/BlockDGMSmall.png)

**Dataset**: The project use the dataset **[DataGTZAN music/speech collection](http://opihi.cs.uvic.ca/sound/music_speech.tar.gz)**. All the wav audio files should be extracted to the *Data/Files* folder.

Install dependency libraries using virtual environment from the **requirements** file

**Excecute the following Jupyter Notebooks/Scripts in order**

	 1. dataProcessing.py
	 2. dlModeling.py
	 3. realTimeTest.py or realTimeTest.ipynb



