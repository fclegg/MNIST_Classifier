import os

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from matplotlib.pyplot import imshow
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

train_batch_size = 64
test_batch_size = 1000
NET_PATH = './MNIST_Net.pth'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


def main():
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    training_data = datasets.MNIST(root="data", download=True, train=True, transform=transform)
    testing_data = datasets.MNIST(root="data", download=True, train=False, transform=transform)

    train_loader = DataLoader(training_data, train_batch_size) #maybe add num_workers or shuffle later?
    test_loader = DataLoader(testing_data, test_batch_size)

    for epoch in range(4):
        if not (os.path.exists(NET_PATH)):
            train(net, train_loader, optimizer, criterion, epoch)
    test(test_loader)


def train(net, train_loader, optimizer, criterion, epoch):
    net.train()

    print("training!")
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_idx % 100 == 0:  # print every 2000 mini-batches
            print(batch_idx)
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    print('Finished Training!')
    torch.save(net.state_dict(), NET_PATH)

def test(test_loader):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # print('GroundTruth: ', ' '.join(f'{labels[j]:5s}' for j in range(4)))

    print('GroundTruth: ')
    for j in range(4):
        print(str(labels[j]), end=", ")

    net = Net()
    net.load_state_dict(torch.load(NET_PATH))

    outputs = net(images)
    predicted = torch.max(outputs, 1)

    #print('Predicted: ', ' '.join(f'{predicted[j]:5s}' for j in range(4)))
    print('\nPredicted: ')
    for j in range(4):
        print(str(predicted[1][j]), end=", ")
    misses = 0
    for i in range(len(predicted[1])):

        if predicted[1][i] != labels[i]:
            print(predicted[1][i] == labels[i], str(predicted[1][i]), str(labels[i]), "index: ", i, end = "\n")

            misses +=1


    print("misses: ", misses / len(predicted[1]))
    while True:
        z = int(input(""))
        image = images[z]
        image = np.array(image, dtype= 'float')
        pixels = image.reshape((28,28))
        plt.imshow(pixels,cmap='gray')
        plt.show()
main()