import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

max_len = 200
x_train = pad_sequences(
    x_train, maxlen=max_len, padding='post', truncating='post'
)
x_test = pad_sequences(
    x_test, maxlen=max_len, padding='post', truncating='post'
)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)


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
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


EPOCHS = 10
BATCH_SIZE = 64

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    model.fit(x_train, y_train, validation_split=0.2, epochs=1,
              batch_size=BATCH_SIZE, verbose=1)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy after epoch {epoch}: {acc:.4f}, loss: {loss:.4f}\n")
