from torch.utils import data
import numpy as np
import torchvision
import torch

from Transforms import build_transform

__all__ = [
    'build_dataset_CIFAR10',
    'build_dataset_STL10',
    'build_dataset_ImageNet10',
    'build_dataset_ImageNetDogs',
]

def get_dataloader(train_dataset, test_dataset, args):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    args.test_size = len(test_dataset)
    return train_dataloader, test_dataloader


def build_dataset_CIFAR10(dataset_path, args):
    transform_train = build_transform(True, args)
    transform_test = build_transform(False, args)

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=False,
        transform=transform_test
    )
    
    train_dataloader, test_dataloader = get_dataloader(train_dataset, test_dataset, args)
    return train_dataloader, test_dataloader


def build_dataset_STL10(dataset_path, args):
    transform_train = build_transform(True, args)
    transform_test = build_transform(False, args)

    train_dataset = torchvision.datasets.STL10(
        root=dataset_path,
        split='train',
        transform=transform_train
    )

    test_dataset = torchvision.datasets.STL10(
        root=dataset_path,
        split='test',
        transform=transform_test
    )
    
    train_dataloader, test_dataloader = get_dataloader(train_dataset, test_dataset, args)
    return train_dataloader, test_dataloader


def build_dataset_ImageNet10(dataset_path, args):
    transform_train = build_transform(True, args)
    transform_test = build_transform(False, args)

    train_dataset = torchvision.datasets.ImageFolder(
        root=dataset_path,
        transform=transform_train,
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=dataset_path,
        transform=transform_test,
    )

    train_dataloader, test_dataloader = get_dataloader(train_dataset, test_dataset, args)
    return train_dataloader, test_dataloader


def build_dataset_ImageNetDogs(dataset_path, args):
    transform_train = build_transform(True, args)
    transform_test = build_transform(False, args)

    train_dataset = torchvision.datasets.ImageFolder(
        root=dataset_path,
        transform=transform_train,
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=dataset_path,
        transform=transform_test,
    )

    train_dataloader, test_dataloader = get_dataloader(train_dataset, test_dataset, args)
    return train_dataloader, test_dataloader