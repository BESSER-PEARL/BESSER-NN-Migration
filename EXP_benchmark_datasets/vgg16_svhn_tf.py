import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from tensorflow.keras import layers, Sequential

SEED = 42
BATCH_SIZE = 256
IMG_SIZE = 224
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Mean = [0.43756306, 0.44365236, 0.47271287]
Std = [0.19829315, 0.20127417, 0.19727801]


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - Mean) / Std
    return image, label

ds_train = tfds.load('svhn_cropped', split='train', as_supervised=True)
ds_train = ds_train.map(preprocess).shuffle(10000, seed=SEED)
ds_train = ds_train.batch(BATCH_SIZE)

ds_test = tfds.load('svhn_cropped', split='test', as_supervised=True)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE)



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
        self.p1 = layers.GlobalAveragePooling2D()
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


device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
with tf.device(device_name):
    model_tf = NeuralNetwork()
    model_tf.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    for epoch in range(EPOCHS):
        history = model_tf.fit(ds_train, epochs=1, validation_data=ds_test)
        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0]
        val_loss = history.history['val_loss'][0]
        val_acc = history.history['val_accuracy'][0]
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Test Loss={val_loss:.4f}, Test Acc={val_acc:.4f}")
