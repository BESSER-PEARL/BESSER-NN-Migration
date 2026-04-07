import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

x_train = torch.tensor(x_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

batch_size = 64
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


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
optimizer = optim.Adam(model.parameters())

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

    # Evaluate on test set
    model.eval()
    correct, total = 0, 0
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            test_loss += loss.item() * xb.size(0)
            predicted = outputs.argmax(dim=1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    test_acc = correct / total
    test_loss /= total
    print(f"Epoch {epoch}: Test Accuracy: {test_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}")
