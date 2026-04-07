import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


(train_data, test_data), info = tfds.load(
    'svhn_cropped', split=['train', 'test'], as_supervised=True,
    with_info=True
)


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_data = train_data.map(preprocess)
test_data = test_data.map(preprocess)

train_images = []
train_labels = []
for img, lbl in tfds.as_numpy(train_data):
    train_images.append(img)
    train_labels.append(lbl)
train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)

test_images = []
test_labels = []
for img, lbl in tfds.as_numpy(test_data):
    test_images.append(img)
    test_labels.append(lbl)
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)


train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(in_channels=3, out_channels=32,
                            kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.actv_func_relu = nn.ReLU()
        self.l2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.l3 = nn.Conv2d(in_channels=32, out_channels=64,
                            kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.l4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.l5 = nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.l6 = nn.Flatten(start_dim=1, end_dim=-1)
        self.l7 = nn.Linear(in_features=1024, out_features=64)
        self.l8 = nn.Linear(in_features=64, out_features=10)


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.l1(x)
        x = self.actv_func_relu(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.actv_func_relu(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.actv_func_relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.l6(x)
        x = self.l7(x)
        x = self.actv_func_relu(x)
        x = self.l8(x)
        return x




model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {correct/total:.4f}")