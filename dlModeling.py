# https://www.kdnuggets.com/2020/07/getting-started-tensorflow2.html
# https://github.com/tensorflow/tensorflow/issues/21738
# https://stats.stackexchange.com/questions/272607/cifar-10-cant-get-above-60-accuracy-keras-with-tensorflow-backend

# Import Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.initializers import RandomUniform
import keras
import matplotlib.pyplot as plt
from tensorflow import feature_column

# set values for clean data visualization
labelsize 	= 12
width 		= 5
height 		= width / 1.618

plt.rc('font', family ='serif')
plt.rc('text', usetex = True)
plt.rc('xtick', labelsize = labelsize)
plt.rc('ytick', labelsize = labelsize)
plt.rc('axes', labelsize = labelsize)




print('TF Version is: ', tf.version.VERSION)

# Import Refined DataSet
datasrc  = 'MasterData.csv'

essentialdata = pd.read_csv('Data/'+datasrc, low_memory=False).dropna()



features = essentialdata.drop(['speechpath','speaker_gender'], axis=1).values
labels = essentialdata.loc[:,['speaker_gender']].values


print('Shape of Input: ', features.shape)
print('Shape of Labels: ', labels.shape)

# classinfo = np.unique(labels.flatten(), return_counts=True)[1]
# a = classinfo[0]
# b = classinfo[1]

# print('Category 1: {}%'.format(np.round(a*100/(a+b),2)))
# print('Category 2: {}%'.format(np.round(b*100/(a+b),2)))

# Perform Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle = True)



def create_model():

	model = Sequential()

	model.add(Dense(64, activation = 'relu', input_shape= X_train[0].shape))
	model.add(Dense(512, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05),
		kernel_regularizer=tf.keras.regularizers.l2(0.001),
	 	activation=tf.nn.relu))
	model.add(Dense(128, activation = 'sigmoid', 
		kernel_regularizer=tf.keras.regularizers.l2(0.001)))
	model.add(Dense (128, activation = 'relu',
		kernel_regularizer=tf.keras.regularizers.l2(0.001)))
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(Dense (64, activation = 'sigmoid', 
		kernel_regularizer=tf.keras.regularizers.l2(0.001)))
	model.add(Dense(2, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05),
		kernel_regularizer=tf.keras.regularizers.l2(0.001),
	 	activation=tf.nn.softmax))



	model.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	return model


# Create a basic model instance
model = create_model()


checkpoint_path = "./TrainedModel/"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)




# Train the model with callback
history = model.fit(X_train, 
	y_train, 
	batch_size = 32,
	epochs=200, 
	validation_split = 0.25,
	callbacks=[cp_callback])


fig1, ax 	= plt.subplots()
fig1.subplots_adjust(left=.16, bottom=.2, right=.99, top=.97)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper left')
fig1.set_size_inches(width, height)
plt.savefig('Graphs/Train_Valiation_Loss.png', dpi=300)
plt.close()

fig2, ax 	= plt.subplots()
fig1.subplots_adjust(left=.16, bottom=.2, right=.99, top=.97)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper right')
fig2.set_size_inches(width, height)
plt.savefig('Graphs/Train_Valiation_Accuracy.png', dpi=300)
plt.close()