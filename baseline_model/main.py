import os
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from operator import itemgetter
import torch
from torch import nn, optim
import numpy as np
import re
from PIL import Image
import random


dataset_path = "../Datasets/MelanomaDetection/"
train_dataset_path = dataset_path + "labeled"
test_dataset_path = dataset_path + "test"


def data_loader(batch_size, train_transform, test_transform):
    train_dataset = MelanomaDataset(extract_label, train_dataset_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MelanomaDataset(extract_label, test_dataset_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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
            # image = io.read_image(full_img_name)
            # image = image.float()

            if self.transform:
                image = self.transform(image)

            result = {'name': img_name,
                      'image': image}

            if self.label_extractor:
                result['label'] = self.label_extractor(img_name)

            return result


def validate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = itemgetter('image', 'label')(data)
            outputs = model(images)
            predicted = outputs.apply_(lambda e: 1 if e > 0.5 else 0)
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

    return correct / total, correct, total


def train(model, criterion, train_loader, test_loader, lr, epochs, momentum):
    # Each iteration of the loader serves up a pair (images, labels)
    # The images are [64, 1, 28, 28] and the labels [64]
    # The batch size is 64 images and the images are 28 x 28.
    losses = []
    test_accuracies = []

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for e in range(epochs):
        print("\nEpocs: ", e + 1)
        model.train()
        running_loss = 0
        for data in train_loader:
            images, labels = itemgetter('image', 'label')(data)

            # zeros all the gradients of the weights
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.float().unsqueeze(1))

            # Calculates all the gradients via backpropagation
            loss.backward()

            # Adjust weights based on the gradients
            optimizer.step()

            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        test_accuracy, test_correct, test_total = validate(model, test_loader)
        print("Loss: ", loss)
        print("Test accuracy:", test_accuracy, ", Correct: ", test_correct, ", Total:", test_total)
        losses.append(loss)
        test_accuracies.append(test_accuracy)

    return losses, test_accuracies


def create_basic_model():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(3, 24, (3, 3))
            self.mp = nn.MaxPool2d((2, 2))
            self.conv2 = nn.Conv2d(24, 48, (3, 3))
            self.flatten = nn.Flatten()
            self.re = nn.ReLU()
            self.l1 = nn.Linear(1728, 28)
            self.dropout = nn.Dropout(0.5)
            self.l2 = nn.Linear(28, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.conv1(x)
            x = self.re(x)
            x = self.mp(x)

            x = self.conv2(x)
            x = self.re(x)
            x = self.mp(x)

            x = self.flatten(x)
            x = self.l1(x)
            x = self.dropout(x)
            x = self.l2(x)
            x = self.sigmoid(x)

            return x

    return Model()


def create_trained_model():
    class PretrainedModel(nn.Module):
        def __init__(self):
            super(PretrainedModel, self).__init__()
            self.resnet = models.resnet18(pretrained=True)
            for param in self.resnet.parameters():
                param.requires_grad = False

            self.linear = nn.Linear(1000, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.resnet(x)
            x = self.linear(x)
            x = self.sigmoid(x)
            return x

    return PretrainedModel()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def scale_image(image):
    return image * 256


def run_basic_model(batch_size):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(12321)

    lr = 0.001
    momentum = 0
    epochs = 10

    transform = transforms.Compose([transforms.ToTensor(), scale_image])
    train_loader, test_loader = data_loader(batch_size, transform, transform)
    criterion = nn.BCELoss()
    model = create_basic_model()

    _, test_error = train(model, criterion, train_loader, test_loader, lr, epochs, momentum)

    print()
    print("Highest test accuracy:", max(test_error))
    print("Number of epocs:", np.argmax(test_error) + 1)

    # Highest test accuracy: 72%
    # Continuing to train wil take the train accuracy up to 100%, but the test accuracy
    # does not get any better


def augmentation_transforms():
    rotation = transforms.RandomChoice(
        [transforms.RandomRotation([-3, 3]),
         transforms.RandomRotation([87, 93]),
         transforms.RandomRotation([177, 183]),
         transforms.RandomRotation([267, 273])])

    return transforms.Compose([transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               rotation])


def run_augmented_model(batch_size):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(12321)

    lr = 0.001
    momentum = 0.2
    epochs = 120

    base_transform = transforms.Compose([transforms.ToTensor(), scale_image])
    augmentation = augmentation_transforms()
    preprocess = transforms.Compose([base_transform, augmentation])

    train_loader, test_loader = data_loader(batch_size, preprocess, base_transform)

    criterion = nn.BCELoss()
    model = create_basic_model()

    _, test_error = train(model, criterion, train_loader, test_loader, lr, epochs, momentum)

    print()
    print("Highest test accuracy:", max(test_error))
    print("Number of epocs:", np.argmax(test_error) + 1)

    # Highest test accuracy: 76%
    # Also a lot of sensitivity to initial (ie seed). Some seeds, will
    # fail to train
    # Takes a lot longer to train - but a 4% increase in accuracy


def run_pretrained_model(batch_size):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(1)

    base = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
    augmentation = augmentation_transforms()
    preprocess = transforms.Compose([base, augmentation])

    train_loader, test_loader = data_loader(batch_size, preprocess, base)

    criterion = nn.BCELoss()
    model = create_trained_model()

    lr = 0.001
    momentum = 0.9
    epochs = 3

    _, test_error = train(model, criterion, train_loader, test_loader, lr, epochs, momentum)

    print()
    print("Highest test accuracy:", max(test_error))
    print("Number of epocs:", np.argmax(test_error) + 1)


def main():
    batch_size = 32

    run_basic_model(batch_size)
    run_augmented_model(batch_size)
    run_pretrained_model(batch_size)


if __name__ == '__main__':
    main()


