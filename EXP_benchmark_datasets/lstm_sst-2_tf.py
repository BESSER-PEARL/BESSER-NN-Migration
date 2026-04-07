import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
import numpy as np

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

VOCAB_SIZE = 10000
MAX_LEN = 50

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

X_train = pad_sequences(
    tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding='post'
)
X_val = pad_sequences(
    tokenizer.texts_to_sequences(val_texts), maxlen=MAX_LEN, padding='post'
)

y_train = tf.keras.utils.to_categorical(train_labels, num_classes=2)
y_val = tf.keras.utils.to_categorical(val_labels, num_classes=2)


class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = layers.Embedding(input_dim=10000, output_dim=326)
        self.l2 = layers.Bidirectional(layers.LSTM(units=40, dropout=0.5,
                                                   return_sequences=True))
        self.l3 = layers.Dropout(rate=0.2)
        self.l4 = layers.LSTM(units=40, dropout=0.2)
        self.l5 = layers.Dense(units=40, activation='relu')
        self.l6 = layers.Dense(units=2, activation='softmax')

        
    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x
    

model = NeuralNetwork()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS = 50
BATCH_SIZE = 64

for epoch in range(EPOCHS):
    history = model.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE,
                        verbose=0)
    train_loss = history.history['loss'][0]
    train_acc = history.history['accuracy'][0]

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Test Loss={val_loss:.4f}, Test Acc={val_acc:.4f}")
