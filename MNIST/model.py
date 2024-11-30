import torch
import torch.nn as nn
import ssl
import os

from configuration.init_config import DEVICE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set the SSL path
ssl._create_default_https_context = ssl._create_unverified_context

def prepare_data(batch_size):
    # Transform and normalizate data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_path = "/Users/faustyn/Developer/STU_2324/UI/Zadanie3a/data"
    train_path = os.path.join(data_path, 'raw/train-images-idx3-ubyte')
    test_path = os.path.join(data_path, 'raw/t10k-images-idx3-ubyte')

    download = not (os.path.exists(train_path) and os.path.exists(test_path))

    # Load train and test data
    train_dataset = datasets.MNIST(root=data_path, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class MNISTClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate):
        super(MNISTClassifier, self).__init__()
        layers = []
        prev_size = input_size

        # Dynamic create hidden layerss
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        return self.network(x)

def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    return total_loss / len(test_loader), 100. * correct / total