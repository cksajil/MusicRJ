import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.initializers import RandomUniform


datasrc  = 'MasterData.csv'
essentialdata = pd.read_csv('Data/'+datasrc, low_memory=False).dropna()
picktoTest = essentialdata.sample(n=1)


def create_model():

	model = Sequential()

	model.add(Dense(64, activation = 'relu', input_shape= (1, 256)))
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


# Load Saved Model
modeltoDeploy = create_model()


# Loads the weights
checkpoint_path = "./TrainedModel/"
checkpoint_dir = os.path.dirname(checkpoint_path)
modeltoDeploy.load_weights(checkpoint_path)


classes = ['Female', 'Male']


# Re-evaluate the model
# loss, acc = modeltoDeploy.evaluate(X_test, y_test, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
# prediction = modeltoDeploy.predict(X_test[:1])
# print(prediction)

print(picktoTest)
testData = picktoTest.drop(['speechpath','speaker_gender'], axis=1).values


probability_model = Sequential([modeltoDeploy, tf.keras.layers.Softmax()])
predictions = probability_model.predict(testData)
index = np.argmax(predictions[0])

print(index)
print(classes[index])