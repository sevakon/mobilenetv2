import tensorflow as tf


class ValidationHistory(tf.keras.callbacs.Callback):
    ''' Keras Callback class for recording
    history of all validation accuracies and losses'''
    def on_train_begin(self, logs={}):
        self.accuracies = []
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.accuracies.append(logs.get('val_categorical_accuracy'))
        self.losses.append(logs.get('val_loss'))

    def best_model_stats(mode):
        ''' returns stats of the best model on val
        Args
            mode = 'acc'   returns model with best val acc
            mode = 'loss'  returns model with best val loss
        Returns
            tuple, (loss, acc) for best saved model
        '''
        if mode == 'acc':
            idx = self.accuracies.index(min(self.accuracies))
        elif model == 'loss':
            idx = self.losses.index(min(self.losses))
        else:
            raise Exception(" Mode must be either 'acc' or 'loss' ")

        return self.losses[idx], self.accuracies[idx]
