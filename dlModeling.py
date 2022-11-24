# Import libraries
import os
import keras
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.initializers import RandomUniform

# Set values for clean data visualization
labelsize = 12
width = 5
height = width / 1.618

plt.rc('font', family = 'serif')
plt.rc('text', usetex = True)
plt.rc('xtick', labelsize = labelsize)
plt.rc('ytick', labelsize = labelsize)
plt.rc('axes', labelsize = labelsize)

# Import refined dataSet
datasrc = 'MasterData.csv'
essentialdata = pd.read_csv('./Data/'+datasrc,low_memory=False).dropna()
essentialdata = essentialdata.replace({'Speech': 0, 'Music': 1})
essentialdata = essentialdata.sample(frac=1)
features = essentialdata.drop(['Filename', 'Label'], axis=1).values
labels = essentialdata.loc[:,['Label']].values

# Perform Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle = True)

# Define the Neural Network Model and return it
def create_model():
	model = Sequential()
	model.add(Dense(64,\
		activation = 'relu',\
		input_shape = X_train[0].shape))
	model.add(Dense(512,\
		kernel_initializer = RandomUniform(minval=-0.05, maxval=0.05),\
		kernel_regularizer = tf.keras.regularizers.l2(0.001),\
		activation = tf.nn.relu))
	model.add(Dense(128,\
		activation = 'relu',\
		kernel_regularizer = tf.keras.regularizers.l2(0.001)))
	model.add(Dense(128,\
		activation = 'relu',\
		kernel_regularizer = tf.keras.regularizers.l2(0.001)))
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(Dense(64,\
		activation = 'relu',\
		kernel_regularizer = tf.keras.regularizers.l2(0.001)))
	model.add(Dense(2,\
		kernel_initializer = RandomUniform(minval=-0.05, maxval=0.05),\
		kernel_regularizer = tf.keras.regularizers.l2(0.001),\
	 	activation = tf.nn.softmax))
	model.compile(optimizer = keras.optimizers.RMSprop(learning_rate=0.0001),\
		loss='sparse_categorical_crossentropy',\
		metrics=['accuracy'])
	return model

# Create a basic model instance
model = create_model()

# Give path to the location where trained model is to be saved
checkpoint_path = "./TrainedModel/"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,\
                                                 save_weights_only=True,\
                                                 verbose=1)

# Train the model with callback
history = model.fit(X_train,\
	y_train,\
	batch_size = 32,\
	epochs = 20,\
	validation_split = 0.20,\
	callbacks=[cp_callback])

# Load Saved Model
modeltoDeploy = create_model()

# Loads the weights
checkpoint_path = "./TrainedModel/"
checkpoint_dir = os.path.dirname(checkpoint_path)
modeltoDeploy.load_weights(checkpoint_path)
ndatapoints = features.shape[0]

# Re-evaluate the model
loss, acc = modeltoDeploy.evaluate(X_test, y_test, verbose=2)
stats = "Restored model, accuracy: {:5.2f}\% on {} data points".format(100 * acc, ndatapoints)
print(stats)

# Plot the loss graph
fig1, ax = plt.subplots()
fig1.subplots_adjust(left=.16, bottom=.2, right=.99, top=.90)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(stats)
plt.legend(['Training', 'Validation'], loc='upper left')
fig1.set_size_inches(width, height)
plt.savefig('./Graphs/Train_Valiation_Loss_'+str(ndatapoints)+'.png', dpi=300)
plt.close()

# Plot the accuracy graph
fig2, ax = plt.subplots()
fig1.subplots_adjust(left=.16, bottom=.2, right=.99, top=.97)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(stats)
plt.legend(['Training', 'Validation'], loc = 'upper right')
fig2.set_size_inches(width, height)
plt.savefig('./Graphs/Train_Valiation_Accuracy_'+str(ndatapoints)+'.png', dpi=300)
plt.close()
