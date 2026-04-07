import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

max_len = 200
x_train = pad_sequences(
    x_train, maxlen=max_len, padding='post',truncating='post'
)
x_test = pad_sequences(
    x_test, maxlen=max_len, padding='post', truncating='post'
)

x_train = torch.tensor(x_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.float32)

BATCH_SIZE = 256
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


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
optimizer = optim.Adam(model.parameters())


EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            test_loss += loss.item() * xb.size(0)
            predicted = (outputs >= 0.5).float()
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    test_acc = correct / total
    test_loss /= total
    print(f"Epoch {epoch}: Test Accuracy: {test_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}")
