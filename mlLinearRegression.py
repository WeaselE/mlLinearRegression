from tensorflow import model
from tensorflow import Adam
from tensorflow import SGD
from tensorflow import keras
# import pandas as pd
i# mport matplotlib.pyplot as plt
from IPython.display import clear_output
# from six.moves import urllib
# import tensorboard
# from datetime import datetime

import numpy as np

#model design for neural network
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Define the Keras TensorBoard callback.
# logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#loss method of understanding the results of each epoch, and optimizer for understanding what to try next to optimize next result.
model.compile(optimizer="SGD", loss="MeanSquaredError")
Adam = tf.keras.optimizers.SGD(learning_rate=1000000)
#model data for training
xs = np.array([-5.0, -4.0, -3.0,-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
ys = np.array([-11.0, -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)

#training loop for model, figure out how to fit x values into y values
model.fit(xs, ys, epochs=1000, validation_split=0.2, verbose=1, callbacks=[tensorboard_callback])

#prediction should be or close to 19
print(model.predict([10.0]))