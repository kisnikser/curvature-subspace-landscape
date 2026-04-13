import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_mnist_dataset(root, train=True):
    """Return MNIST dataset (without DataLoader)."""
    transform = get_mnist_transform()
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    return dataset


def get_mnist_loader(root, batch_size, train=True, seed=None):
    transform = get_mnist_transform()
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, generator=generator)
    return loader
