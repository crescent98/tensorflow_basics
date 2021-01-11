from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model

from model import u_net_output

if __name__ == "__main__":
    # Set some parameters
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    # COMPILE THE MODEL
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    output = u_net_output(inputs)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
