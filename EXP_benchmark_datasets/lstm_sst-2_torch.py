import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

y_train = np.array(train_labels, dtype=np.int64)
y_val = np.array(val_labels, dtype=np.int64)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long),
                              torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long),
                            torch.tensor(y_val, dtype=torch.long))

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Embedding(num_embeddings=10000, embedding_dim=326)
        self.l2 = nn.LSTM(input_size=326, hidden_size=40, bidirectional=True,
                          dropout=0.5, batch_first=True)
        self.l3 = nn.Dropout(p=0.2)
        self.l4 = nn.LSTM(input_size=80, hidden_size=40, bidirectional=False,
                          dropout=0.2, batch_first=True)
        self.l5 = nn.Linear(in_features=40, out_features=40)
        self.actv_func_relu = nn.ReLU()
        self.l6 = nn.Linear(in_features=40, out_features=2)
        self.actv_func_softmax = nn.Softmax()


    def forward(self, x):
        x = self.l1(x)
        x, _ = self.l2(x)
        x = self.l3(x)
        x, _ = self.l4(x)
        x = x[:, -1, :]
        x = self.l5(x)
        x = self.actv_func_relu(x)
        x = self.l6(x)
        x = self.actv_func_softmax(x)
        return x


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    val_loss /= total
    val_acc = correct / total

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
