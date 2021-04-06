import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

data_set_path = '../Datasets'

z_dims = 100
image_size = 28 * 28


def set_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(12321)


def data_loader(batch_size):
    # transform = transforms.ToTensor()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST(data_set_path, download=False, train=True, transform=transform)
    test_set = datasets.MNIST(data_set_path, download=False, train=False, transform=transform)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_generator_model():
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            # z_dims
            self.fc1 = nn.Linear(z_dims, 256 * (7 * 7))
            # 256 x (7 x 7)
            self.cvt1 = nn.ConvTranspose2d(256, 128, (3, 3), stride=2, padding=1, output_padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(128)
            self.leakyReLU1 = nn.ReLU()
            # 128 x (14 x 14)
            self.cvt2 = nn.ConvTranspose2d(128, 64, (3, 3), stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.leakyReLU2 = nn.ReLU()
            # 64 x (14 x 14)
            self.cvt3 = nn.ConvTranspose2d(64, 1, (3, 3), stride=2, padding=1, output_padding=1, bias=False)
            # 1 x (28 x 28)

        def forward(self, x):
            x = self.fc1(x)
            x = x.view(x.size(0), 256, 7, 7)
            x = self.cvt1(x)
            x = self.bn1(x)
            x = self.leakyReLU1(x)
            x = self.cvt2(x)
            x = self.bn2(x)
            x = self.leakyReLU2(x)
            x = self.cvt3(x)
            x = torch.tanh(x)

            return x

    return Generator()


# LeakyRU 0.01 OR 0.2

def create_new_discriminator_model():
    # 1 means a fake
    # 0 means real
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            # 1 x (28 x 28)
            self.cnv1 = nn.Conv2d(1, 64, (4, 4), stride=2, padding=1)
            self.leakyReLU1 = nn.LeakyReLU(0.2)
            # 64 x (14 x 14)
            self.cnv2 = nn.Conv2d(64, 64, (4, 4), stride=2, padding=1)
            self.leakyReLU2 = nn.LeakyReLU(0.2)
            # 64 x (7 x 7)
            self.fc1 = nn.Linear(64 * 7 * 7, 1)

        def forward(self, x):
            x = self.cnv1(x)
            x = self.leakyReLU1(x)
            x = self.cnv2(x)
            x = self.leakyReLU2(x)
            x = torch.flatten(x, 1, -1)
            x = self.fc1(x)
            x = torch.sigmoid(x)

            return x

    return Discriminator()


def create_linear_discriminator_model():
    # 1 means a fake
    # 0 means real
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.fc1 = nn.Linear(image_size, 400)
            self.fc2 = nn.Linear(400, 1)

        def forward(self, x):
            x = torch.flatten(x, 1, -1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = torch.sigmoid(x)

            return x

    return Discriminator()


def create_old_discriminator_model():
    # 1 means a fake
    # 0 means real
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            # 1 x (28 x 28)
            self.cnv1 = nn.Conv2d(1, 32, (3, 3), stride=2, padding=1)
            self.leakyReLU1 = nn.LeakyReLU(0.2)
            # 32 x (14 x 14)
            self.cnv2 = nn.Conv2d(32, 64, (3, 3), stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.leakyReLU2 = nn.LeakyReLU(0.2)
            # 64 x (7 x 7)
            self.cnv3 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=0)
            self.bn3 = nn.BatchNorm2d(128)
            self.leakyReLU3 = nn.LeakyReLU(0.2)
            # 128 x (3 x 3)
            self.fc1 = nn.Linear(128 * 3 * 3, 1)

        def forward(self, x):
            x = self.cnv1(x)
            x = self.leakyReLU1(x)

            x = self.cnv2(x)
            x = self.bn2(x)
            x = self.leakyReLU2(x)

            x = self.cnv3(x)
            x = self.bn3(x)
            x = self.leakyReLU3(x)

            x = torch.flatten(x, 1, -1)
            x = self.fc1(x)
            x = torch.sigmoid(x)

            return x

    return Discriminator()


def batch_random_noise(batch_size):
    return torch.randn(batch_size, z_dims)


def view_images(epoch, generator, discriminator, sample):
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        # obtain one batch of test images
        gen_image = generator(sample)
        classification = discriminator(gen_image)
        gen_image = gen_image.view(sample.size(0), 1, 28, 28)
        gen_image = 0.5 * gen_image + 0.5
        classification = classification.squeeze()

        # plot the first ten input images, then reconstructed images, then generated images
        fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True)

        # input images on top row, reconstructions on bottom
        # for images, row, title in zip([gen_image], axes, ['Generated']):
        # axes[0].set_title('Epoch: ' + str(epoch))

        fig.suptitle('Epoch: ' + str(epoch))

        for img, label, ax in zip(gen_image, classification, axes):
            img = img.detach().numpy()
            ax.set_title('Fake' if label > 0.5 else 'Real')
            ax.imshow(np.squeeze(img), cmap='gray_r')

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


def gen_images(epoch, index, generator, sample):
    with torch.no_grad():
        generator.eval()
        fake_images = generator(sample)
        fake_images *= -1
        file_name = 'results/gen_' + str(epoch) + "_" + str(index) + ".png"
        save_image(fake_images, file_name, normalize=True)


def accuracy(output, wanted):
    with torch.no_grad():
        predicted = output.numpy() > 0.5
        total = output.size(0)
        correct = (predicted == wanted.numpy()).sum().item()
        return correct / total


def train(sample, epochs, train_loader, generator_optimizer, discriminator_optimizer,
          generator, discriminator, criterion):
    batches_per_epoch = 100

    discriminator.train()
    generator.train()

    # Don't use eval for discriminator when training the generator - and Batch Normalisation
    # It is a complete disaster.
    # Instead keep the discriminator in training mode
    for epoch in range(epochs):
        print("\nEpoch: ", epoch)
        # for index, (images, labels) in enumerate(train_loader):
        data_iter = iter(train_loader)
        for index in range(batches_per_epoch):
            images, labels = data_iter.next()

            batch_size = labels.size(0)
            # real_ones = torch.zeros(batch_size)
            # fakes = torch.ones(batch_size)

            # Train discriminator - real batch
            # Discriminator must mark the real images 0 and the fake images 1
            discriminator_optimizer.zero_grad()
            output_real = discriminator(images)
            loss_real = criterion(output_real.squeeze(), torch.zeros(batch_size))
            d_real_accuracy = accuracy(output_real.squeeze(), torch.zeros(batch_size))
            loss_real.backward()

            # Train discriminator - fake batch
            random_noise1 = batch_random_noise(batch_size)
            fake_images1 = generator(random_noise1)
            output_fake1 = discriminator(fake_images1.detach())
            loss_fake = criterion(output_fake1.squeeze(), torch.ones(batch_size))
            d_loss_accuracy = accuracy(output_fake1.squeeze(), torch.ones(batch_size))
            loss_fake.backward()
            discriminator_optimizer.step()

            # Train generator
            generator_optimizer.zero_grad()

            # random_noise2 = batch_random_noise(batch_size)
            # fake_images2 = generator(random_noise2)
            output_fake2 = discriminator(fake_images1)

            # Generator must cause the discriminator to mark the fake images as real
            # ie to fool the discriminator
            generator_loss = criterion(output_fake2.squeeze(), torch.zeros(batch_size))
            gen_accuracy = accuracy(output_fake2.squeeze(), torch.zeros(batch_size))

            generator_loss.backward()
            generator_optimizer.step()

            generator_total_loss = generator_loss.item()
            discriminator_loss_real = loss_real.item()
            discriminator_loss_fake = loss_fake.item()

            print('Epoch: %d, Iteration: %i, Gen Loss: %.3f, Gen Accuracy: %.3f, '
                  'DReal loss: %.3f, DReal accuracy: %.3f, '
                  'DFake loss: %.3f, DFake accuracy: %.3f' %
                  (epoch, index, generator_total_loss, gen_accuracy,
                   discriminator_loss_real, d_real_accuracy,
                   discriminator_loss_fake, d_loss_accuracy))

            if (index % 50) == 0:
                gen_images(epoch, index, generator, sample)


def weights_init(model):
    # Specific initialisation of weights needed for Conv, Conv transpose
    # and BatchNorm layers
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def main():
    set_seeds()

    epochs = 80
    lr = 0.0002
    beta1 = 0.5
    batch_size = 128
    sample = torch.randn(batch_size, z_dims)

    train_loader, test_loader = data_loader(batch_size)
    generator = create_generator_model()
    generator.apply(weights_init)

    discriminator = create_old_discriminator_model()
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()

    generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    train(sample, epochs, train_loader, generator_optimizer, discriminator_optimizer,
          generator, discriminator, criterion)

    view_images(epochs, generator, discriminator, sample)
    plt.show()


main()
