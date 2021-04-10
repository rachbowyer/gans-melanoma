import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import os
from PIL import Image
import random
import re
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image


dataset_path = "../Datasets/MelanomaDetection/"
train_dataset_path = dataset_path + "labeled"
test_dataset_path = dataset_path + "test"
unlabeled_dataset_path = dataset_path + "unlabeled"


z_dims = 20
image_size = 32 * 32


def set_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(12321)


def weights_init(model):
    # Specific initialisation of weights needed for Conv, Conv transpose
    # and BatchNorm layers
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def array_to_dictionary(array):
    return {k: v for k, v in enumerate(array)}


def extract_label(s):
    if re.findall(".*_1.jpg", s):
        return 1
    elif re.findall(".*_0.jpg", s):
        return 0
    else:
        raise RuntimeError("Invalid filename format: " + s)


class MelanomaDataset(Dataset):
    """Unlabelled Melanoma datasets"""

    def __init__(self, label_extractor, dir_path, transform=None):
        self.label_extractor = label_extractor
        self.dir_path = dir_path
        self.transform = transform
        file_list = filter(lambda e: e != ".DS_Store", os.listdir(dir_path))
        self.file_list = array_to_dictionary(file_list)
        self.len = len(self.file_list)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index >= self.len:
            raise IndexError
        else:
            img_name = self.file_list[index]
            full_img_name = os.path.join(self.dir_path, img_name)
            image = Image.open(full_img_name)

            if self.transform:
                image = self.transform(image)

            result = {'name': img_name,
                      'image': image}

            if self.label_extractor:
                result['label'] = self.label_extractor(img_name)

            return result


def get_splits(dataset, percentage_train):
    len_train_set = len(dataset)
    train_set = int(len_train_set*percentage_train)
    val_set = len_train_set - train_set
    return train_set, val_set


def data_loader(batch_size, train_transform, test_transform):
    train_dataset = MelanomaDataset(extract_label, train_dataset_path, transform=train_transform)
    train_set_len, val_set_len = get_splits(train_dataset, 0.7)
    train_dataset, val_dataset = data.random_split(train_dataset, [train_set_len, val_set_len])
    test_dataset = MelanomaDataset(extract_label, test_dataset_path, transform=test_transform)
    unlabeled_dataset = MelanomaDataset(None, unlabeled_dataset_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, unlabeled_loader, val_loader, test_loader


def batch_random_noise(batch_size):
    return torch.randn(batch_size, z_dims, 1, 1)


def augmentation_transforms():
    rotation = transforms.RandomChoice(
        [transforms.RandomRotation([-3, 3]),
         transforms.RandomRotation([87, 93]),
         transforms.RandomRotation([177, 183]),
         transforms.RandomRotation([267, 273])])

    return transforms.Compose([transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               rotation])


# def create_simple_generator_model():
#     class Generator(nn.Module):
#         def __init__(self):
#             super(Generator, self).__init__()
#             # 100
#             self.fc1 = nn.Linear(z_dims, 400)
#             # 400
#             self.fc2 = nn.Linear(400, 3 * image_size)
#             # 3 * 32 * 32
#
#         def forward(self, x):
#             x = x.flatten(1, -1)
#             x = self.fc1(x)
#             x = F.relu(x)
#             x = self.fc2(x)
#             x = torch.sigmoid(x)
#             x = x.view(x.size(0), 3, 32, 32)
#             return x
#
#     return Generator()


def create_generator_model():
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            # z_dims x 1 x 1
            self.cvt1 = nn.ConvTranspose2d(z_dims, 64 * 8, 4, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(64 * 8)
            self.reLU1 = nn.ReLU(True)
            # 512 x (4 x 4)
            self.cvt2 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(64 * 4)
            self.reLU2 = nn.ReLU(True)
            # 256 x (8 x 8)
            self.cvt3 = nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(64 * 2)
            self.reLU3 = nn.ReLU(True)
            # 128 x (16 x 16)
            self.cvt4 = nn.ConvTranspose2d(64 * 2, 3, 4, 2, 1)
            # 3 x (32 x 32)

        def forward(self, x):
            x = self.cvt1(x)
            x = self.bn1(x)
            x = self.reLU1(x)
            x = self.cvt2(x)
            x = self.bn2(x)
            x = self.reLU2(x)
            x = self.cvt3(x)
            x = self.bn3(x)
            x = self.reLU3(x)
            x = self.cvt4(x)
            # x = torch.tanh(x)
            x = torch.sigmoid(x)
            return x

    return Generator()


# def create_simple_discriminator_model():
#     # falsification_neuron - 1 means real, 0 means a fake
#     # prognosis_neuron - 1 means real, 0 means fake
#     class Discriminator(nn.Module):
#         def __init__(self):
#             super(Discriminator, self).__init__()
#             self.fc1 = nn.Linear(3 * image_size, 400)
#             self.falsification_neuron = nn.Linear(400, 1)
#             self.prognosis_neuron = nn.Linear(400, 1)
#
#         def forward(self, x):
#             x = x.flatten(1, -1)
#             x = self.fc1(x)
#             x = F.relu(x)
#             falsification = self.falsification_neuron(x)
#             falsification = torch.sigmoid(falsification)
#             prognosis = self.prognosis_neuron(x)
#             prognosis = torch.sigmoid(prognosis)
#
#             return falsification, prognosis
#
#     return Discriminator()


def create_discriminator_model():
    # Neuron 0 means negative and real
    # Neuron 1 means positive and real

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            # 3 x (32 x 32)
            self.cnv1 = nn.Conv2d(3, 64, 4, 1, 2, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.reLU1 = nn.LeakyReLU(0.2, inplace=True)
            # 64 x (33 x 33)
            self.cnv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.reLU2 = nn.LeakyReLU(0.2, inplace=True)
            # 128 x (16 x 16)
            self.cnv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(256)
            self.reLU3 = nn.LeakyReLU(0.2, inplace=True)
            # 256 x (8 x 8)
            self.cnv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
            self.bn4 = nn.BatchNorm2d(512)
            self.reLU4 = nn.LeakyReLU(0.2, inplace=True)
            # 512 x (4 x 4)
            # self.fc = nn.Linear(512 * 4 * 4, 2)
            self.output_neurons = nn.Conv2d(512, 2, 4, 1, 0)
            # self.output_neurons = nn.Conv2d(512, 1, 4, 1, 0)

        def forward(self, x):
            x = self.cnv1(x)
            x = self.bn1(x)
            x = self.reLU1(x)
            x = self.cnv2(x)
            x = self.bn2(x)
            x = self.reLU2(x)
            x = self.cnv3(x)
            x = self.bn3(x)
            x = self.reLU3(x)
            x = self.cnv4(x)
            x = self.bn4(x)
            x = self.reLU4(x)
            x = self.output_neurons(x)
            x = x.flatten(1, -1)
            # x = self.fc(x)
            # x = torch.sigmoid(x)
            return x

    return Model()


def gen_images(epoch, index, generator, sample):
    with torch.no_grad():
        # generator.eval()
        fake_images = generator(sample)
        file_name = 'results/gen_' + str(epoch) + "_" + str(index) + ".png"
        save_image(fake_images, file_name, normalize=True)


def view_images(epoch, generator, discriminator, sample):
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        # obtain one batch of test images
        gen_image = generator(sample)
        classification = discriminator(gen_image)
        gen_image = gen_image.view(sample.size(0), 3, 32, 32)
        gen_image = 0.5 * gen_image + 0.5
        classification = classification.squeeze()

        # plot the first ten input images, then reconstructed images, then generated images
        fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True)

        # input images on top row, reconstructions on bottom
        # for images, row, title in zip([gen_image], axes, ['Generated']):
        # axes[0].set_title('Epoch: ' + str(epoch))

        fig.suptitle('Epoch: ' + str(epoch))

        for img, label, ax in zip(gen_image, classification, axes):
            ax.set_title('Real' if label > 0.5 else 'Fake')
            adj_image = torch.swapaxes(torch.swapaxes(img, 0, 1), 1, 2)

            # Format for imshow - (M, N, 3)
            ax.imshow(adj_image)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


def accuracy_fn_labeled(output, wanted):
    with torch.no_grad():
        predicted = torch.argmax(output.data, 1)
        total = output.size(0)
        correct = (predicted == wanted).sum().item()
        return correct / total


def accuracy_fn_fake(output, want_real):
    # Determine if there is at least a 50% chance that
    # these are real moles (or at least a 50% chance that they are fake)
    with torch.no_grad():
        t = torch.exp(torch.logsumexp(output, dim=1))
        probability = t / (t+1) if want_real else 1 / (t+1)
        predicted = probability.numpy() > 0.5
        total = output.size(0)
        correct = predicted.sum().item()
        return correct / total


def real_loss(output):
    lse = torch.logsumexp(output, dim=1)
    # lse_plus_one = F.softplus(torch.exp(lse))
    lse_plus_one = torch.log(torch.exp(lse) + 1)
    return (-lse + lse_plus_one).mean()


def fake_loss(output):
    lse = torch.logsumexp(output, dim=1)
    # lse_plus_one = F.softplus(torch.exp(lse))
    lse_plus_one = torch.log(torch.exp(lse) + 1)
    return lse_plus_one.mean()


def model_accuracy(discriminator, a_data_loader, accuracy):
    with torch.no_grad():
        running_accuracy = 0
        for some_data in a_data_loader:
            images, labels = itemgetter('image', 'label')(some_data)
            output = discriminator(images)
            acc = accuracy(output, labels)
            running_accuracy += acc

    return running_accuracy / len(a_data_loader)


def train(sample, epochs, train_loader, unlabeled_loader, val_loader, test_loader,
          generator_optimizer, discriminator_optimizer,
          generator, discriminator, train_discriminator):
    batches_per_epoch = len(unlabeled_loader)
    # batches_per_epoch = 2
    print("Batches per epoc: ", batches_per_epoch)

    discriminator.train()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()

    if generator is not None:
        generator.train()

    # Don't use eval for discriminator when training the generator - and Batch Normalisation
    # It is a complete disaster.
    # Instead keep the discriminator in training mode
    total_count = 0
    unlabeled_loader_iter = None

    for epoch in range(epochs):
        print("\nEpoch: ", epoch)
        running_accuracy = 0
        running_dreal_accuracy = 0

        for index in range(batches_per_epoch):
            # Load batches
            if (total_count % len(unlabeled_loader)) == 0:
                unlabeled_loader_iter = iter(unlabeled_loader)

            unlabeled_images = unlabeled_loader_iter.next()['image']

            if (total_count % len(train_loader)) == 0:
                labeled_loader_iter = iter(train_loader)

            labeled_data = labeled_loader_iter.next()
            images, labels = itemgetter('image', 'label')(labeled_data)

            # Train discriminator - labeled batch
            if train_discriminator:
                discriminator_optimizer.zero_grad()
                output = discriminator(images)

                loss = criterion(output, labels)

                labeled_accuracy = accuracy_fn_labeled(output, labels)
                running_accuracy += labeled_accuracy
                loss.backward()
                discriminator_optimizer.step()

                print('Epoch: %d, Iteration: %i, Labelled Loss: %.3f' %
                      (epoch, index, loss.item()))

            # Train discriminator - unlabeled batch
            if generator is not None:
                unlabeled_batch_size = unlabeled_images.size(0)
                # real_ones = torch.full((unlabeled_batch_size,), 1.0, dtype=torch.float)
                # fakes = torch.zeros(unlabeled_batch_size)

                # Train discriminator - real batch
                # Discriminator must mark the real images 1 and the fake images 0
                discriminator_optimizer.zero_grad()
                output_real = discriminator(unlabeled_images)

                loss_real = real_loss(output_real)
                d_real_accuracy = accuracy_fn_fake(output_real, want_real=True)
                running_dreal_accuracy += d_real_accuracy
                loss_real.backward()

                # Train discriminator - fake batch
                random_noise = batch_random_noise(unlabeled_batch_size)
                fake_images = generator(random_noise)
                output_fake_1 = discriminator(fake_images.detach())
                loss_fake = fake_loss(output_fake_1)
                d_loss_accuracy = accuracy_fn_fake(output_fake_1, want_real=False)
                loss_fake.backward()
                discriminator_optimizer.step()

                # Train generator
                generator_optimizer.zero_grad()
                output_fake_2 = discriminator(fake_images)

                # Generator must cause the discriminator to mark the fake images as real
                # ie to fool the discriminator
                generator_loss = real_loss(output_fake_2)
                gen_accuracy = accuracy_fn_fake(output_fake_2, want_real=True)
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

                if (total_count % 20) == 0:
                    gen_images(epoch, index, generator, sample)

            total_count += 1
        running_accuracy /= batches_per_epoch
        running_dreal_accuracy /= batches_per_epoch
        validation_accuracy = model_accuracy(discriminator, val_loader, accuracy_fn_labeled)

        print("End of epoc")
        print("Training accuracy in spotting fakes for epoc: ", running_dreal_accuracy)
        print("Training accuracy in discriminating malignant/benign for epoc: ", running_accuracy)
        print("Validation accuracy in discriminating malignant/benign for epoc: ", validation_accuracy)
        test_accuracy = model_accuracy(discriminator, test_loader, accuracy_fn_labeled)
        print("Test accuracy in discriminating malignant/benign: ", test_accuracy)


def main():
    set_seeds()

    epochs = 50
    lr = 0.0002
    beta1 = 0.5
    batch_size = 128
    sample = batch_random_noise(batch_size)

    base = transforms.ToTensor()
    augmentation = augmentation_transforms()
    preprocess = transforms.Compose([base, augmentation])

    train_loader, unlabeled_loader, val_loader, test_loader \
        = data_loader(batch_size, preprocess, base)

    generator = create_generator_model()
    generator.apply(weights_init)

    discriminator = create_discriminator_model()
    discriminator.apply(weights_init)

    generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    train(sample, epochs, train_loader, unlabeled_loader, val_loader, test_loader,
          generator_optimizer, discriminator_optimizer,
          generator, discriminator, train_discriminator=True)

    gen_images(epochs, 0, generator, sample)

    #
    # test_accuracy = model_accuracy(discriminator, test_loader)
    # print("Test accuracy: ", test_accuracy)

    # view_images(epochs, generator, discriminator, sample)
    # plt.show()


if __name__ == '__main__':
    main()
