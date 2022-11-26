# Import libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from plotter.plots import qualityLinePlot
from utils.general import create_model

# Import refined dataSet
data_src = 'MasterData.csv'
essentialdata = pd.read_csv('./Data/'+data_src, low_memory=False).dropna()
essentialdata = essentialdata.replace({'Speech': 0, 'Music': 1})
essentialdata = essentialdata.sample(frac=1)
features = essentialdata.drop(['Filename', 'Label'], axis=1).values
labels = essentialdata.loc[:, ['Label']].values

# Perform Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, shuffle=True
)

# Create a basic model instance
model = create_model()

# Give path to the location where trained model is to be saved
checkpoint_path = "./TrainedModel/"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

# Train the model with callback
history = model.fit(
    X_train, y_train, batch_size=32, epochs=20, validation_split=0.20,
    callbacks=[cp_callback])

# Load Saved Model
modeltoDeploy = create_model()

# Loads the weights
checkpoint_path = "./TrainedModel/"
checkpoint_dir = os.path.dirname(checkpoint_path)
modeltoDeploy.load_weights(checkpoint_path)
ndatapoints = features.shape[0]

# Re-evaluate the model
loss, test_accuracy = modeltoDeploy.evaluate(X_test, y_test, verbose=2)
stats = "Restored model, accuracy: {:5.2f}% on {} data points".format(
    100 * test_accuracy, ndatapoints)	
print(stats)

# Plot the loss graph
qualityLinePlot(history.history['loss'])

# Plot the accuracy graph
# qualityLinePlot(history.history['accuracy'])
