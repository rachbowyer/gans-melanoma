#
# Implementation a CNN that classifies the MNIST dataset
# My goal was a CNN that was simple yet had a high degree of accuracy
# The topology is based on an example from a MITx course
# Code is heavily commented with my notes on how the algorithm/Torch is working
#

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn, optim
import numpy as np
data_set_path = '../Datasets'


def data_loader(batch_size):
    # ToTensor - Converts a PIL Image to a tensor
    # Normalize - normalizes the data. The first parameter is the desired
    # mean. And the second parameter the desired standard deviation
    # Both the mean and standard deviation are vectors - each element
    # of the vector corresponds to one "channel"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # Dataset is hosted on a private website. And the website is down
    # So instead always rely on local copy - downloaded from GitHub mirror https://github.com/mkolod/MNIST
    # http://yann.lecun.com/exdb/mnist/
    # Images are black and white 28 x 28 pixel images
    # 60,000 images in the training set. 10,000 images in the test set
    # Images are returned as PIL images
    # Train - loads training database

    train_set = datasets.MNIST(data_set_path, download=False, train=True, transform=transform)
    # Trainset contains targets - a vector of 60,000 values and data [60000, 28, 28]

    test_set = datasets.MNIST(data_set_path, download=False, train=False, transform=transform)
    # Testset contains targets - a vector of 10,000 values and data [10000, 28, 28]

    # Loaders are iterables over datasets
    # Batch size is the number of rows per iteration
    # The standard gradient descent algorithm looks at the error of all data points, and steps
    # based on the derivative of the total error.
    # The stochastic gradient descent (SGD) looks at the error of a batch of points and steps
    # based on the derivative of that partial error. The batch is chosen at random

    # The loaders support SGD by batching and randomising the order of batches

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def create_model():
    # RELU - rectifier linear unit - max(0, x) - is used on the hidden layers
    # LogSoftmax is used on the output layer.
    # Softmax converts the output layer into probabilities proportional to the
    # exponent of the input numbers
    # LogSoftMax - is the log of this
    # The model output model(images) - is a tensor that contains both
    # the output batch size x 10.
    return nn.Sequential(
              nn.Conv2d(1, 32, (3, 3)),
              # Parameters 32 * 9 = 288
              # Output 32 images that are 26 x 26
              nn.ReLU(),
              nn.MaxPool2d((2, 2)),
              # Output 32 images that are 13 x 13

              nn.Conv2d(32, 64, (3, 3)),
              # Parameters 64 * 9 = 576
              # Output 64 images that are 11 x 11
              nn.ReLU(),
              nn.MaxPool2d((2, 2)),
              # Output 64 images that are 5 x 5 = 1600

              nn.Flatten(),
              nn.Linear(1600, 128),
              nn.Dropout(0.5),
              nn.Linear(128, 10),
              nn.LogSoftmax(dim=1)
            )


def loss_function():
    # Use the negative log likelihood loss function
    # Loss function is a measure of the error - like mean squared error
    # or hinge loss (used for SVM)
    # Negative log likelihood needs log probabilities which LogSoftMax serves up
    # Rather than train our model on a binary classification - right/wrong - instead
    # we are training our model to give accurate probabilities of the chance of success
    # Cross entropy loss is the same as negative log likelihood, but the soft max layer
    # is taken care of automatically
    # Confident but wrong predictions are heavily penalised
    return nn.NLLLoss()


def validate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def train(model, criterion, train_loader, test_loader, lr, epochs, momentum):
    # Each iteration of the loader serves up a pair (images, labels)
    # The images are [64, 1, 28, 28] and the labels [64]
    # The batch size is 64 images and the images are 28 x 28.
    losses = []
    test_errors = []

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            # zeros all the gradients of the weights
            optimizer.zero_grad()

            print(images)
            output = model(images)
            loss = criterion(output, labels)

            # Calculates all the gradients via backpropagation
            loss.backward()

            # Adjust weighs based on the gradients
            optimizer.step()

            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        test_error = validate(model, test_loader)

        print("\nEpocs: ", e + 1)
        print("Loss: ", loss)
        print("Test error: ", test_error)
        losses.append(loss)
        test_errors.append(test_error)

    return losses, test_errors


def get_misclassified(model, test_loader):
    result_images = []
    result_labels = []
    result_probs = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)

            for img, label, output, predict in zip(images, labels, outputs, predicted):
                if label != predict:
                    result_images.append(img.squeeze(0))
                    result_labels.append(label)
                    result_probs.append(torch.exp(output).numpy())

    return result_images, result_labels, result_probs


def show_misclassified(images, labels, probs):
    # Shows the first 60 misclassified images and also prints their classification
    # probabilities. Allows to see visually how difficult the images are

    num_of_images = min(60, len(images))

    np.set_printoptions(suppress=True)
    print("For each misclassified image, here are the classification probabilities")
    for i in range(num_of_images):
        print("Label: ", labels[i])
        print("Probabilities: ", probs[i])

    figure = plt.figure()
    for index in range(num_of_images):
        plt.subplot(6, 10, index + 1)
        plt.axis('off')
        plt.imshow(images[index], cmap='gray_r')
    plt.show()


def show_loss_errors(losses, test_errors):
    n = len(losses)
    x_axis = range(1, n+1)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Model performance with increased epocs')

    ax1.plot(x_axis, losses, 'o-')
    ax1.set_ylabel('Training loss')
    ax1.xaxis.set_visible(False)

    ax2.plot(x_axis, test_errors, 'o-')
    ax2.set_ylabel('Test accuracy')

    ax2.set_xlabel('Epocs')
    plt.xticks(x_axis)

    plt.show()


def main():
    # For consistent results
    torch.manual_seed(12321)

    # Hyper parameters
    batch_size = 1
    epochs = 20
    lr = 0.003
    momentum = 0.9

    train_loader, test_loader = data_loader(batch_size)
    criterion = loss_function()

    # Train and validate model
    model = create_model()
    losses, test_error = train(model, criterion, train_loader, test_loader, lr, epochs, momentum)
    show_loss_errors(losses, test_error)

    print("Lowest test error:", max(test_error))
    print("Number of epocs:", np.argmax(test_error) + 1)

    # Show misclassified
    # images, labels, probs = get_misclassified(model, test_loader)
    # show_misclassified(images, labels, probs)


main()




