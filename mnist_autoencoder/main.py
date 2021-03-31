from torchvision import datasets, transforms
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import random

#  https://medium.com/analytics-vidhya/dimension-manipulation-using-autoencoder-in-pytorch-on-mnist-dataset-7454578b018


def train(model, criterion, train_loader, lr, epochs):
    # Each iteration of the loader serves up a pair (images, labels)
    # The images are [32, 1, 28, 28] and the labels [64]
    # The batch size is 32 images and the images are 28 x 28.
    losses = []

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        print("\nEpocs: ", e + 1)
        model.train()
        running_loss = 0
        for images, _ in train_loader:
            # Flatten images - flattened images go in and come out of the
            # network
            images = images.view(images.size(0), -1)

            # zeros all the gradients of the weights
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, images)

            # Calculates all the gradients via backpropagation
            loss.backward()

            # Adjust weights based on the gradients
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        loss = running_loss / len(train_loader)
        print("Loss: ", loss)

        losses.append(loss)



# Simple linear model. Drops input town to encoding_dim
# before expanding back up to full size
# Sigmoid output layer to force into range (-1, 1)
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, encoding_dim)
        self.fc2 = nn.Linear(encoding_dim, 28 * 28)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(12321)

    data_set_path = '../Datasets'
    batch_size = 20

    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize((0.5,), (0.5,)),
                                    ])

    train_set = datasets.MNIST(data_set_path, download=False, train=True, transform=transform)
    test_set = datasets.MNIST(data_set_path, download=False, train=False, transform=transform)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # # obtain one batch of training images
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # images = images.numpy()
    #
    # # get one image from the batch
    # img = np.squeeze(images[0])
    #
    # fig = plt.figure(figsize = (5,5))
    # ax = fig.add_subplot(111)
    # ax.imshow(img, cmap='gray')
    # plt.show()

    encoding_dim = 10
    model = Autoencoder(encoding_dim)

    # Simple mean square loss - similar to regression
    # Not classifying, so not interested in probability based loss measures such as
    # SoftMax or cross entropy loss
    criterion = nn.MSELoss()

    lr = 0.001
    epochs = 5
    train(model, criterion, train_loader, lr, epochs)



    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    images_flatten = images.view(images.size(0), -1)
    # get sample outputs
    output = model(images_flatten)
    # prep images for display
    images = images.numpy()

    # output is resized into a batch of images
    output = output.view(batch_size, 1, 28, 28)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

if __name__ == '__main__':
    main()
