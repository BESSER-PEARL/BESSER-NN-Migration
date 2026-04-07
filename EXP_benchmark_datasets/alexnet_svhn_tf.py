import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, Model, Sequential

SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 10
LR = 1e-4
DEVICE = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"


def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

ds_train = tfds.load('svhn_cropped', split='train', as_supervised=True)
ds_train = ds_train.map(preprocess).shuffle(1000, seed=SEED)
ds_train = ds_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_test = tfds.load('svhn_cropped', split='test', as_supervised=True)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

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
        self.p1 = layers.GlobalAveragePooling2D()
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
    model = NeuralNetwork()
    optimizer = tf.keras.optimizers.Adam(LR)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in ds_train:
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_fn(labels, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss += loss.numpy() * images.shape[0]
            train_correct += tf.reduce_sum(
                tf.cast(tf.argmax(logits, axis=1) == labels, tf.float32)
            ).numpy()
            train_total += images.shape[0]

        train_loss /= train_total
        train_acc = train_correct / train_total

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        for images, labels in ds_test:
            logits = model(images, training=False)
            loss = loss_fn(labels, logits)
            val_loss += loss.numpy() * images.shape[0]
            val_correct += tf.reduce_sum(
                tf.cast(tf.argmax(logits, axis=1) == labels, tf.float32)
            ).numpy()
            val_total += images.shape[0]

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Test Loss={val_loss:.4f}, Test Acc={val_acc:.4f}")
