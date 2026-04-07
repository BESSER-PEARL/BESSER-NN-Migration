import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch import nn, optim

SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

ds_train = tfds.load('svhn_cropped', split='train', as_supervised=True)
ds_train = ds_train.map(preprocess).shuffle(1000, seed=SEED)
ds_train = ds_train.batch(BATCH_SIZE).prefetch(1)

ds_test = tfds.load('svhn_cropped', split='test', as_supervised=True)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(1)


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        x = x.permute(self.dims)
        return x

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            Permute(dims=[0, 3, 1, 2]),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11),
                      stride=(4, 4), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(5, 5),
                      stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0),
            Permute(dims=[0, 2, 3, 1]),
        )
        self.p1 = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.f1 = nn.Flatten(start_dim=1, end_dim=-1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=10),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 3, 1, 2)
        x = self.p1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.f1(x)
        x = self.classifier(x)
        return x


model = NeuralNetwork().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for batch in tfds.as_numpy(ds_train):
        images, labels = batch
        images = torch.from_numpy(images).to(DEVICE)
        labels = torch.from_numpy(labels).long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += images.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for batch in tfds.as_numpy(ds_test):
            images, labels = batch
            images = torch.from_numpy(images).to(DEVICE)
            labels = torch.from_numpy(labels).long().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += images.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Test Loss={val_loss:.4f}, Test Acc={val_acc:.4f}")