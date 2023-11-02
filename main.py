import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

train_batch_size = 64;
test_batch_size = 1000;

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28-6) * (28-6), 10)

        )

    def forward(self, x):
        return self.model(x)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    training_data = datasets.MNIST(root="data", download=True, train=True, transform=transform)
    testing_data = datasets.MNIST(root="data", download=True, train=False, transform=transform)

    train_loader = DataLoader(training_data, train_batch_size) #maybe add num_workers or shuffle later?
    test_loader = DataLoader(testing_data, test_batch_size)


