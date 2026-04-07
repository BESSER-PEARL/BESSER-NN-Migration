import torch
from torch import nn
from torchvision import datasets, transforms
from sklearn.metrics import classification_report 


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

def load_and_preprocess_data(train_path, test_path, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
		transforms.ToTensor()
        ])
    train_dataset = datasets.ImageFolder(
        root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(
        root=test_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        total_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 200 == 199:
                print(
                    f"[{epoch + 1}, {i + 1:5d}] "
                    f"loss: {running_loss / 200:.3f}"
                )
                running_loss = 0.0
        print(
            f"[{epoch + 1}] overall loss for epoch: "
            f"{total_loss / len(train_loader):.3f}"
        )
    print('Training finished')

def evaluate_model(model, test_loader, criterion):
    with torch.no_grad():
        predicted_labels = []
        true_labels = []
        test_loss = 0.0
        for data in test_loader:
            inputs, labels = data
            true_labels.extend(labels)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted)
            test_loss += criterion(outputs, labels).item()

    average_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_loss:.3f}")

    metrics = ['f1-score']
    report = classification_report(true_labels, predicted_labels, 
                                   output_dict=True)
    for metric in metrics:
        metric_list = []
        for class_label in report.keys():
            if class_label not in ('macro avg', 'weighted avg', 'accuracy'):
                print(f"{metric.capitalize()} for class {class_label}:",
                    report[class_label][metric])
                metric_list.append(report[class_label][metric])
        metric_value = sum(metric_list) / len(metric_list)
        print(f"Average {metric.capitalize()}: {metric_value:.2f}")
        print(f"Accuracy: {report['accuracy']}")


def main():
    train_path = "dataset/cifar10/train"
    test_path = "dataset/cifar10/test"
    batch_size = 32
    epochs = 10
    image_size = (32, 32)

    train_loader, test_loader = load_and_preprocess_data(
        train_path, test_path, image_size, batch_size
    )
    my_model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
    train_model(my_model, train_loader, criterion, optimizer, epochs)
    evaluate_model(my_model, test_loader, criterion)



if __name__ == "__main__":
    main()