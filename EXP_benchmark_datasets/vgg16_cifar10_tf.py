import random
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import numpy as np


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)



device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print("TF device:", device_name)

(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train','test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

IMG_SIZE = 224
BATCH_SIZE = 64

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)




def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

ds_train = ds_train.shuffle(500, seed=SEED).batch(BATCH_SIZE)
ds_train = ds_train.map(
    lambda x, y: (tf.image.resize(x, [IMG_SIZE, IMG_SIZE]), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(1)


ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.map(
    lambda x, y: (tf.image.resize(x, [IMG_SIZE, IMG_SIZE]), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(1)


class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.features = Sequential([
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                             padding='valid'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                             padding='valid'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                             padding='valid'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                             padding='valid'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                          padding='valid', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                             padding='valid'),
        ])
        self.p1 = tfa.layers.AdaptiveAveragePooling2D(output_size=(7, 7))
        self.f1 = layers.Flatten()

        self.classifier = Sequential([
            layers.Dense(units=4096, activation='relu'),
            layers.Dropout(rate=0.5),
            layers.Dense(units=4096, activation='relu'),
            layers.Dropout(rate=0.5),
            layers.Dense(units=10),
        ])


    def call(self, x):
        x = self.features(x)
        x = self.p1(x)
        x = self.f1(x)
        x = self.classifier(x)
        return x


with tf.device(device_name):
    model_tf = NeuralNetwork()
    model_tf.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    EPOCHS = 10
    for epoch in range(EPOCHS):
        history = model_tf.fit(ds_train, epochs=1, validation_data=ds_test,
                               verbose=1)
        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0]
        val_loss = history.history['val_loss'][0]
        val_acc = history.history['val_accuracy'][0]
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Test Loss={val_loss:.4f}, Test Acc={val_acc:.4f}")
