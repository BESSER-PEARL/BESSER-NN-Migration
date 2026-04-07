import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np



class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = layers.Embedding(input_dim=5000, output_dim=50)
        self.l2 = layers.Dropout(rate=0.5)
        self.l3 = layers.Conv1D(filters=200, kernel_size=4, strides=1,
                                padding='valid', activation='relu')
        self.l4 = layers.MaxPool1D(pool_size=2, strides=2, padding='valid')
        self.l5 = layers.Conv1D(filters=200, kernel_size=5, strides=1,
                                padding='valid', activation='relu')
        self.l6 = layers.MaxPool1D(pool_size=2, strides=2, padding='valid')
        self.l7 = layers.Dropout(rate=0.15)
        self.l8 = layers.GRU(units=100, dropout=0.0)
        self.l9 = layers.Dense(units=400, activation='relu')
        self.l10 = layers.Dropout(rate=0.1)
        self.l11 = layers.Dense(units=1, activation='sigmoid')

        
    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x_1 = self.l3(x)
        x_1 = self.l4(x_1)
        x_2 = self.l5(x)
        x_2 = self.l6(x_2)
        x_2 = tf.concat([x_1, x_2], axis=-1)
        x_2 = self.l7(x_2)
        x_2 = self.l8(x_2)
        x_2 = self.l9(x_2)
        x_2 = self.l10(x_2)
        x_2 = self.l11(x_2)
        return x_2


ds_train, ds_val = tfds.load('glue/sst2', split=['train', 'validation'])

def extract_text_label(example):
    return example['sentence'], example['label']

ds_train = ds_train.map(extract_text_label)
ds_val = ds_val.map(extract_text_label)

train_texts, train_labels = [], []
for text, label in tfds.as_numpy(ds_train):
    train_texts.append(text.decode('utf-8'))
    train_labels.append(label)

val_texts, val_labels = [], []
for text, label in tfds.as_numpy(ds_val):
    val_texts.append(text.decode('utf-8'))
    val_labels.append(label)

MAX_LEN = 50
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

X_train = pad_sequences(
    tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding='post'
)
X_val = pad_sequences(
    tokenizer.texts_to_sequences(val_texts), maxlen=MAX_LEN, padding='post'
)

y_train = np.array(train_labels, dtype="float32")
y_val = np.array(val_labels, dtype="float32")


model = NeuralNetwork()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

EPOCHS = 50
BATCH_SIZE = 64

for epoch in range(EPOCHS):
    history = model.fit(
        X_train, y_train,
        epochs=1,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    train_loss = history.history['loss'][0]
    train_acc = history.history['accuracy'][0]

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Test Loss={val_loss:.4f}, Test Acc={val_acc:.4f}")
