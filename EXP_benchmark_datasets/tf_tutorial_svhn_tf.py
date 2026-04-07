import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers


(train_data, test_data), info = tfds.load(
    'svhn_cropped', split=['train', 'test'], as_supervised=True, 
    with_info=True
)


def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_data = train_data.map(preprocess_image).batch(64)
train_data = train_data.prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(preprocess_image).batch(64)
test_data = test_data.prefetch(tf.data.AUTOTUNE)

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = layers.Conv2D(filters=32, kernel_size=(3, 3),
                                strides=(1, 1), padding='valid',
                                activation='relu')
        self.l2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                   padding='valid')
        self.l3 = layers.Conv2D(filters=64, kernel_size=(3, 3),
                                strides=(1, 1), padding='valid',
                                activation='relu')
        self.l4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                   padding='valid')
        self.l5 = layers.Conv2D(filters=64, kernel_size=(3, 3),
                                strides=(1, 1), padding='valid',
                                activation='relu')
        self.l6 = layers.Flatten()
        self.l7 = layers.Dense(units=64, activation='relu')
        self.l8 = layers.Dense(units=10, activation=None)


    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        return x


model = NeuralNetwork()

model.compile(
    optimizer='adam', metrics=['accuracy'],
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(train_data, epochs=10, validation_data=test_data)
