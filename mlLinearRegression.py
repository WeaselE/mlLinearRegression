import tensorflow as tf
from tensorflow import keras
import numpy as np

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1)
])

xs = np.array(["2.0","4.0","6.0","8.0","10.0"], np.float64)
print(xs)
ys = np.array(["-2.0","-4.0","-6.0","-8.0","-10.0"], np.float16)
print(ys)

loss_fn = tf.keras.losses.MeanSquaredError()
optim_fn = tf.keras.optimizers.SGD()

model.compile(optimizer=optim_fn, loss=loss_fn)

model.fit(xs, ys, epochs=500, verbose=0, validation_split=0.2)

print(model.predict([1]))