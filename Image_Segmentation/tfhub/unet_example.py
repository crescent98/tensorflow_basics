import sys

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.image as mpimg

model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/nvidia/unet/industrial/class_1/1")
])
model.build((None, 512, 512, 1))
model.summary()
img = mpimg.imread('data/1.png')
img = np.expand_dims(img, axis=2)
img = np.expand_dims(img, axis=0)
img = (img - 0.5) / 0.5
output = model(img)
print(output)