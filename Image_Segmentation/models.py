import tensorflow as tf
from utils import *

def unet_model(output_channels):

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # 이 층들의 활성화를 이용
    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]
    
    layers = [base_model.get_layer(name).output for name in layer_names]
    
    # 특징 추출 모델
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # downsampling
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # upstack
    up_stack = [
        pix2pix.upsample(512, 3),
        pix2pix.upsample(256, 3),
        pix2pix.upsample(128, 3),
        pix2pix.upsample(64, 3)
    ]

    # upsampling by skipping, and then setting
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # last layer
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same'
    )

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)