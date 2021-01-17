import tensorflow as tf
import pathlib

if __name__ == "__main__":
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('/home/shmoon/tensorflow_test/simple_test/data', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    print(data_dir)