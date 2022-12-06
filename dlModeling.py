# Import libraries
from os import path
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from plotter.plots import qualityLinePlot
from utils.general import create_model, load_config
from utils.data_loader import DataLoader


def main():

    config = load_config("my_config.yaml")
    data_loader = DataLoader()
    data_loader.load_data()

    # Perform Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        data_loader.features, data_loader.labels,
        test_size=config["test_size"], shuffle=True
    )
    print(data_loader.features.shape)
    # Create a basic model instance
    model = create_model(X_train[0].shape)

    # Give path to the location where trained model is to be saved
    if config['default_model'] == 'myDNN':
        file_path = path.join(config["model_directory"],
                              config["dnn_model_name"])
    else:
        file_path = path.join(config["model_directory"],
                              config["cnn_model_name"])

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=file_path, monitor='val_accuracy',
                                  verbose=1, save_best_only=True, mode='auto')

    # Train the model with callback
    history = model.fit(
        X_train, y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=config["test_size"],
        callbacks=[cp_callback])

    # Load Saved Model
    modeltoDeploy = create_model(X_train[0].shape)

    # Loads the weights
    modeltoDeploy.load_weights(file_path)
    ndatapoints = data_loader.features.shape[0]

    # Re-evaluate the model
    loss, test_accuracy = modeltoDeploy.evaluate(X_test, y_test, verbose=2)
    stats = "Restored model, accuracy: {:5.2f}% on {} data points".format(
        100 * test_accuracy, ndatapoints)
    print(stats)

    # Plot loss and accuracy graphs
    qualityLinePlot(history)


if __name__ == '__main__':
    main()
