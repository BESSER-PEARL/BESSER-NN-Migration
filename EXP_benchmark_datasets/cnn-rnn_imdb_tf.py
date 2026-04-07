import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)


max_len = 200
x_train = pad_sequences(
    x_train, maxlen=max_len, padding='post', truncating='post'
)
x_test = pad_sequences(
    x_test, maxlen=max_len, padding='post', truncating='post'
)


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


model = NeuralNetwork()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])


EPOCHS = 10
BATCH_SIZE = 256


for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    model.fit(x_train, y_train, validation_split=0.2, epochs=1,
              batch_size=BATCH_SIZE, verbose=1)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy after epoch {epoch}: {acc:.4f}, loss: {loss:.4f}\n")
