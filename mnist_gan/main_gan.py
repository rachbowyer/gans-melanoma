import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms

data_set_path = '../Datasets'

z_dims = 28
image_size = 28 * 28


def set_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(12321)


def data_loader(batch_size):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(data_set_path, download=False, train=True, transform=transform)
    test_set = datasets.MNIST(data_set_path, download=False, train=False, transform=transform)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader,  test_loader


def create_generator_model():
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.fc1 = nn.Linear(z_dims, 400)
            self.fc2 = nn.Linear(400, image_size)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = torch.sigmoid(x)
            return x

    return Generator()


def create_discriminator_model():
    # 1 means a fake
    # 0 means real
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.fc1 = nn.Linear(image_size, 400)
            self.fc2 = nn.Linear(400, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = torch.sigmoid(x)
            return x

    return Discriminator()


def batch_random_noise(batch_size):
    return torch.randn(batch_size, z_dims)


def train(train_loader, generator_optimizer, discriminator_optimizer,
          generator, discriminator, criterion):

    generator_total_loss = 0
    discriminator_total_loss = 0

    for images, labels in train_loader:
        batch_size = labels.size(0)

        # Train discriminator
        generator.eval()
        discriminator.train()
        discriminator_optimizer.zero_grad()

        random_noise1 = batch_random_noise(batch_size)
        fake_images1 = generator(random_noise1)
        output_fake1 = discriminator(fake_images1)

        # Flatten images
        images = images.view(images.size(0), -1)
        output_real = discriminator(images)

        # Discriminator must mark the real images 0 and the fake images 1
        loss_real = criterion(output_real.squeeze(), torch.zeros(batch_size))
        loss_fake = criterion(output_fake1.squeeze(), torch.ones(batch_size))
        loss = loss_real + loss_fake

        loss.backward()
        discriminator_optimizer.step()

        # Train generator
        discriminator.eval()
        generator.train()
        generator_optimizer.zero_grad()

        random_noise2 = batch_random_noise(batch_size)
        fake_images2 = generator(random_noise2)
        output_fake2 = discriminator(fake_images2)

        # Generator must cause the discriminator to mark the fake images 0
        # ie to fool the discriminator
        generator_loss = criterion(output_fake2.squeeze(), torch.zeros(batch_size))

        generator_loss.backward()
        generator_optimizer.step()

        generator_total_loss += generator_loss.item() * batch_size * 2
        discriminator_total_loss += loss.item() * batch_size

    generator_total_loss /= len(train_loader)
    discriminator_total_loss /= len(train_loader)

    print("Generator loss:", generator_total_loss)
    print("Discriminator loss:", discriminator_total_loss)


def view_images(epoch, generator, discriminator, sample):
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        # obtain one batch of test images
        gen_image = generator(sample)
        classification = discriminator(gen_image)
        gen_image = gen_image.view(sample.size(0), 1, 28, 28)
        classification = classification.squeeze()

        # plot the first ten input images, then reconstructed images, then generated images
        fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True)

        # input images on top row, reconstructions on bottom
        # for images, row, title in zip([gen_image], axes, ['Generated']):
        # axes[0].set_title('Epoch: ' + str(epoch))

        fig.suptitle('Epoch: ' + str(epoch + 1))

        for img, label, ax in zip(gen_image, classification,  axes):
            img = img.detach().numpy()
            ax.set_title('Fake' if label > 0.5 else 'Real')
            ax.imshow(np.squeeze(img), cmap='gray_r')

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


def main():
    set_seeds()

    batch_size = 32
    epochs = 5
    lr = 0.001
    sample = torch.randn(batch_size, z_dims)

    train_loader, test_loader = data_loader(batch_size)
    generator = create_generator_model()
    discriminator = create_discriminator_model()
    criterion = nn.BCELoss()

    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    for e in range(epochs):
        print("\nEpocs: ", e + 1)
        train(train_loader, generator_optimizer, discriminator_optimizer,
              generator, discriminator, criterion)

    view_images(epochs, generator, discriminator, sample)

    plt.show()


main()
