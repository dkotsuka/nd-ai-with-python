import torch
from torchvision import datasets, transforms


def get_data_loaders(data_dir):
    """
    Function to get and return train, validation and test data loaders.
    Return:
        train_loader - train data loader
        valid_loader - validation data loader
        test_loader - test data loader
    """

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.RandomRotation(60),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(
        train_datasets, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(
        valid_datasets, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        test_datasets, batch_size=32, shuffle=True)

    return trainloader, validloader, testloader
