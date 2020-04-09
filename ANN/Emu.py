import tensorflow as tf
tf.config.optimizer.set_jit(True)
import numpy as np

class ANN():


    def __int__(self,params):
        self.params = params
        self.params = np.reshape(self.params, (1, 3))

    def model(self,model):

        model = tf.keras.models.load_model(model)

        model = model.predict(self.params)

        return model
