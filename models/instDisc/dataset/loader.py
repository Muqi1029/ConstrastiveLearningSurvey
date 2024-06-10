import torch
from torch import nn
from torch.utils.data import dataset, DataLoader
from torchvision.datasets import CIFAR10
from PIL import Image
import os


class CIFAR10Instance(CIFAR10):
    """Add an index to a CIFAR10 instance
    """

    def __getitem__(self, index: int) -> dataset.Tuple[torch.Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index


def load_dataset(name: str, transform=None, test_transform=None, batch_size: int = 32):
    print(f"pwd: {os.getcwd()}")
    if name == 'cifar10':
        train_dataset = CIFAR10Instance(root="../data",
                                        train=True,
                                        transform=transform,
                                        download=True)
        test_dataset = CIFAR10Instance(root="../data",
                                       train=False,
                                       transform=test_transform,
                                       download=True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise NotImplementedError("Only support CIFAR10Instance")

    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False), classes)
