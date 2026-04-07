import random
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import numpy as np


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


DEVICE = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print("TF device:", DEVICE)


(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train','test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)


IMG_SIZE = 224
BATCH_SIZE = 128

def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_train = ds_train.map(preprocess).shuffle(5000, seed=SEED).batch(BATCH_SIZE)
ds_test  = ds_test.map(preprocess).batch(BATCH_SIZE)


class NeuralNetwork(Model):
    def __init__(self):
        super().__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(padding=2),
            layers.Conv2D(64, (11,11), strides=(4,4), activation='relu'),
            layers.MaxPool2D((3,3), strides=(2,2)),
            layers.ZeroPadding2D(padding=2),
            layers.Conv2D(192, (5,5), activation='relu'),
            layers.MaxPool2D((3,3), strides=(2,2)),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(384, (3,3), activation='relu'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(256, (3,3), activation='relu'),
            layers.ZeroPadding2D(padding=1),
            layers.Conv2D(256, (3,3), activation='relu'),
            layers.MaxPool2D((3,3), strides=(2,2)),
        ])
        self.p1 = tfa.layers.AdaptiveAveragePooling2D((6,6))
        self.f1 = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dense(10)
        ])

    def call(self, x):
        x = self.features(x)
        x = self.p1(x)
        x = self.f1(x)
        x = self.classifier(x)
        return x


with tf.device(DEVICE):
    model_tf = NeuralNetwork()
    model_tf.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    EPOCHS = 10
    for epoch in range(EPOCHS):
        history = model_tf.fit(ds_train, epochs=1, validation_data=ds_test)
        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0]
        val_loss = history.history['val_loss'][0]
        val_acc = history.history['val_accuracy'][0]
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Test Loss={val_loss:.4f}, "
              f"Test Acc={val_acc:.4f}")
    