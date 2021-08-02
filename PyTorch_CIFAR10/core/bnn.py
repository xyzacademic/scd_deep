import tensorflow as tf
import larq as lq
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

# tf.keras.backend.set_image_data_format('channels_first')






class BNN(object):
    def __init__(self, path):
        if isinstance(path, list):
            self.models = [tf.keras.models.load_model(i) for i in path]
            self.best_model = self.models[0]
        elif isinstance(path, str):
            self.best_model = tf.keras.models.load_model(path)
            self.models = [tf.keras.models.load_model(path)]


    def predict(self, data, best_index=None, batch_size=None):
        if best_index is not None:
            yp = self.models[best_index].predict(data).argmax(axis=1)
            return yp
        else:
            if batch_size:
                n_batch = data.shape[0] // batch_size
                n_rest = data.shape[0] % batch_size
                yps = []
                for j in range(n_batch):
                    yp = []
                    for i in range(len(self.models)):
                        yp.append(self.models[i].predict(data[j*batch_size:(j+1)*batch_size]))
                    yp = np.stack(yp, axis=2)
                    yps.append(yp)

                if n_rest > 0:
                    yp = []
                    for i in range(len(self.models)):
                        yp.append(self.models[i].predict(data[n_batch * batch_size:]))
                    yp = np.stack(yp, axis=2)
                    yps.append(yp)
                yp = np.concatenate(yps, axis=0)
            return yp.mean(axis=2).argmax(axis=1)

    def predict_proba(self, data):
        yp = []
        for i in range(len(self.models)):
            yp.append(self.models[i].predict(data))
        yp = np.stack(yp, dim=2)
        return yp.mean(axis=2)