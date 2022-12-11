# Callback code for Early stopping
from tensorflow.keras.callbacks import Callback


class EarlyStopper(Callback):
    """
    A class for early stopper callback for validation accuracy
    """
    def __init__(self, target):
        super(EarlyStopper, self).__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs={}):
        acc = logs['val_accuracy']
        if acc >= self.target:
            self.model.stop_training = True
