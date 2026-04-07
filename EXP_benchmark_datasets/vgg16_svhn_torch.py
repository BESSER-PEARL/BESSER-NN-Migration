import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch import nn, optim
from tqdm import tqdm

SEED = 42
BATCH_SIZE = 256
IMG_SIZE = 224
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Mean = [0.43756306, 0.44365236, 0.47271287]
Std = [0.19829315, 0.20127417, 0.19727801]


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - Mean) / Std
    return image, label

ds_train = tfds.load('svhn_cropped', split='train', as_supervised=True)
ds_train = ds_train.map(preprocess).shuffle(10000, seed=SEED)
ds_train = ds_train.batch(BATCH_SIZE)

ds_test = tfds.load('svhn_cropped', split='test', as_supervised=True)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE)



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
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                      stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            Permute(dims=[0, 2, 3, 1]),
        )
        self.p1 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.f1 = nn.Flatten(start_dim=1, end_dim=-1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
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

    pbar = tqdm(tfds.as_numpy(ds_train), desc=f"Epoch {epoch+1} [Train]",
                leave=False)
    for images, labels in pbar:
        images = torch.from_numpy(images).float().to(DEVICE)
        labels = torch.from_numpy(labels).long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += images.size(0)

        pbar.set_postfix({
            "loss": f"{train_loss/train_total:.4f}",
            "acc": f"{train_correct/train_total:.4f}"
        })

    train_loss /= train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    pbar_val = tqdm(tfds.as_numpy(ds_test), desc=f"Epoch {epoch+1} [Val]",
                    leave=False)
    with torch.no_grad():
        for images, labels in pbar_val:
            
            images = torch.from_numpy(images).float().to(DEVICE)
            labels = torch.from_numpy(labels).long().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += images.size(0)

            pbar_val.set_postfix({
                "loss": f"{val_loss/val_total:.4f}",
                "acc": f"{val_correct/val_total:.4f}"
            })

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Test Loss={val_loss:.4f}, Test Acc={val_acc:.4f}")
