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

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long),
                              torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long),
                            torch.tensor(y_val, dtype=torch.float32))

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Embedding(num_embeddings=5000, embedding_dim=50)
        self.l2 = nn.Dropout(p=0.5)
        self.l3 = nn.Conv1d(in_channels=50, out_channels=200, kernel_size=4,
                            stride=1, padding=0)
        self.actv_func_relu = nn.ReLU()
        self.l4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.l5 = nn.Conv1d(in_channels=50, out_channels=200, kernel_size=5,
                            stride=1, padding=0)
        self.l6 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.l7 = nn.Dropout(p=0.15)
        self.l8 = nn.GRU(input_size=400, hidden_size=100, bidirectional=False,
                         dropout=0.0, batch_first=True)
        self.l9 = nn.Linear(in_features=100, out_features=400)
        self.l10 = nn.Dropout(p=0.1)
        self.l11 = nn.Linear(in_features=400, out_features=1)
        self.actv_func_sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = x.permute(0, 2, 1)
        x_1 = self.l3(x)
        x_1 = self.actv_func_relu(x_1)
        x_1 = self.l4(x_1)
        x_1 = x_1.permute(0, 2, 1)
        x_2 = self.l5(x)
        x_2 = self.actv_func_relu(x_2)
        x_2 = self.l6(x_2)
        x_2 = x_2.permute(0, 2, 1)
        x_2 = torch.cat((x_1, x_2), dim=-1)
        x_2 = self.l7(x_2)
        x_2, _ = self.l8(x_2)
        x_2 = x_2[:, -1, :]
        x_2 = self.l9(x_2)
        x_2 = self.actv_func_relu(x_2)
        x_2 = self.l10(x_2)
        x_2 = self.l11(x_2)
        x_2 = self.actv_func_sigmoid(x_2)
        return x_2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    val_loss /= total
    val_acc = correct / total

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
          f"Train Acc={train_acc:.4f}, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
