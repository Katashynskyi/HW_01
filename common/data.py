import torch
import numpy as np
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset


class GANAugmentedCIFAR10(Dataset):
    """CIFAR-10 for GAN return 1 image"""

    def __init__(
        self,
        root: str,
        train: bool = True,
        aug_pipeline: A.Compose = None,
        normalizer=None,
        multiplier: int = 1,  # Reduced multiplier as GANs need less augmentation
    ):
        self.base_dataset = CIFAR10(root=root, train=train, download=True)
        self.aug_pipeline = aug_pipeline
        self.normalizer = normalizer
        self.multiplier = multiplier

    def __len__(self):
        return len(self.base_dataset) * self.multiplier

    def __getitem__(self, idx):
        true_idx = idx % len(self.base_dataset)
        img, _ = self.base_dataset[true_idx]  # Discard labels

        # Convert to numpy for albumentations
        img = np.array(img)

        # Apply augmentation if available
        if self.aug_pipeline:
            img = self.aug_pipeline(image=img)["image"]

        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Apply normalization if available
        if self.normalizer:
            img = self.normalizer(img)

        return img


class AugmentedCIFAR10(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        aug_pipeline: A.Compose = None,
        normalizer=None,
        multiplier: int = 1,
    ):
        self.base_dataset = CIFAR10(root=root, train=train, download=True)
        self.aug_pipeline = aug_pipeline
        self.normalizer = normalizer
        self.multiplier = multiplier

    def __len__(self):
        return len(self.base_dataset) * self.multiplier

    def __getitem__(self, idx):
        true_idx = idx % len(self.base_dataset)
        img, _ = self.base_dataset[true_idx]

        img = np.array(img)

        if self.aug_pipeline:
            img = self.aug_pipeline(image=img)["image"]

        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        if self.normalizer:
            img = self.normalizer(img)

        return img, img


class VAEDataModule:
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.1,
        augment_multiplier: int = 1,
        subset_size: int = None,  # New parameter
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.augment_multiplier = augment_multiplier
        self.subset_size = subset_size

        self.normalizer = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.train_aug = A.Compose(
            [
                A.OneOf(
                    [
                        A.Sequential(
                            [
                                A.HorizontalFlip(p=0.5),
                                A.ShiftScaleRotate(
                                    shift_limit=0.1,
                                    scale_limit=0.2,
                                    rotate_limit=15,
                                    p=0.5,
                                ),
                                A.ColorJitter(
                                    brightness=0.1,
                                    contrast=0.1,
                                    saturation=0.1,
                                    hue=0.1,
                                    p=0.7,
                                ),
                                A.GaussNoise(var_limit=(0, 0.05 * 255), p=0.3),
                            ],
                            p=0.7,
                        ),
                        A.Sequential(
                            [
                                A.HorizontalFlip(p=0.5),
                                A.RandomBrightnessContrast(
                                    brightness_limit=0.1, contrast_limit=0.1, p=0.7
                                ),
                            ],
                            p=0.3,
                        ),
                    ],
                    p=1.0,
                )
            ]
        )

        self.val_transforms = T.Compose([T.ToTensor(), self.normalizer])

    def setup(self):
        # Create full datasets first
        full_train_dataset = AugmentedCIFAR10(
            root=self.data_dir,
            train=True,
            aug_pipeline=self.train_aug,
            normalizer=self.normalizer,
            multiplier=self.augment_multiplier,
        )

        full_val_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.val_transforms,
            download=True,
        )

        # Apply subset selection if specified
        if self.subset_size is not None:
            indices = torch.randperm(len(full_train_dataset))[: self.subset_size]
            self.train_dataset = torch.utils.data.Subset(full_train_dataset, indices)

            val_size = int(self.subset_size * self.val_split)
            val_indices = torch.randperm(len(full_val_dataset))[:val_size]
            self.val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)

            test_size = val_size
            test_indices = torch.randperm(len(full_val_dataset))[:test_size]
            self.test_dataset = torch.utils.data.Subset(full_val_dataset, test_indices)
        else:
            self.train_dataset = full_train_dataset

            val_size = int(len(full_val_dataset) * self.val_split)
            train_size = len(full_val_dataset) - val_size

            _, self.val_dataset = random_split(
                full_val_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            self.test_dataset = full_val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class GANDataModule:
    """Data module for GAN training on CIFAR-10"""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        augment_multiplier: int = 1,
        subset_size: int = None,  # Added parameter
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_multiplier = augment_multiplier
        self.subset_size = subset_size  # Store the new parameter

        # Initialize normalizer (tanh range)
        self.normalizer = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # Initialize augmentation pipeline - lighter than VAE's
        self.train_aug = A.Compose(
            [
                A.OneOf(
                    [
                        # Moderate augmentation path
                        A.Sequential(
                            [
                                A.HorizontalFlip(p=0.5),
                                A.RandomBrightnessContrast(
                                    brightness_limit=0.1, contrast_limit=0.1, p=0.7
                                ),
                            ],
                            p=0.7,
                        ),
                        # Minimal augmentation path
                        A.Sequential(
                            [
                                A.HorizontalFlip(p=0.5),
                            ],
                            p=0.3,
                        ),
                    ],
                    p=1.0,
                )
            ]
        )

        # Test transforms - just normalization
        self.test_transforms = T.Compose([T.ToTensor(), self.normalizer])

    def setup(self):
        """Setup train and test datasets"""
        # Create full datasets first
        full_train_dataset = GANAugmentedCIFAR10(
            root=self.data_dir,
            train=True,
            aug_pipeline=self.train_aug,
            normalizer=self.normalizer,
            multiplier=self.augment_multiplier,
        )

        full_test_dataset = GANAugmentedCIFAR10(
            root=self.data_dir,
            train=False,
            aug_pipeline=self.train_aug,
            normalizer=self.normalizer,
            multiplier=self.augment_multiplier,
        )

        # Apply subset selection if specified
        if self.subset_size is not None:
            train_indices = torch.randperm(len(full_train_dataset))[: self.subset_size]
            self.train_dataset = torch.utils.data.Subset(
                full_train_dataset, train_indices
            )

            test_size = int(self.subset_size * 0.2)  # Using 20% of subset_size for test
            test_indices = torch.randperm(len(full_test_dataset))[:test_size]
            self.test_dataset = torch.utils.data.Subset(full_test_dataset, test_indices)
        else:
            self.train_dataset = full_train_dataset
            self.test_dataset = full_test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,  # Important for GAN training
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class FlowDataModule:
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        subset_size: int = None,  # New parameter
        valid_split: float = 0.1,  # Added validation split
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.subset_size = subset_size
        self.valid_split = valid_split

        # Normalization transform
        self.normalizer = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # Albumentations pipeline for training
        self.aug_pipeline = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3
                ),
            ]
        )

    def get_subset_indices(self, dataset_size: int, subset_size: int) -> list:
        """Get random subset indices"""
        if subset_size is None or subset_size >= dataset_size:
            return list(range(dataset_size))
        return random.sample(range(dataset_size), subset_size)

    def setup(self):
        """Setup train, validation and test datasets"""
        # Create base CIFAR10 datasets
        full_train_dataset = CIFAR10(root=self.data_dir, train=True, download=True)

        # Create augmented training dataset
        self.train_dataset = AugmentedCIFAR10(
            root=self.data_dir,
            train=True,
            aug_pipeline=self.aug_pipeline,
            normalizer=self.normalizer,
            multiplier=1,
        )

        # Create normalized datasets for validation and test
        self.test_dataset = CIFAR10(
            root=self.data_dir,
            train=False,
            transform=T.Compose([T.ToTensor(), self.normalizer]),
            download=True,
        )

        # Handle subset selection
        if self.subset_size is not None:
            # Get subset indices
            train_valid_indices = self.get_subset_indices(
                len(full_train_dataset), self.subset_size
            )
            test_indices = self.get_subset_indices(
                len(self.test_dataset), self.subset_size
            )

            # Split train indices into train and validation
            num_valid = int(len(train_valid_indices) * self.valid_split)
            valid_indices = train_valid_indices[:num_valid]
            train_indices = train_valid_indices[num_valid:]

            # Create subsets
            self.train_dataset = Subset(self.train_dataset, train_indices)
            self.valid_dataset = Subset(
                CIFAR10(
                    root=self.data_dir,
                    train=True,
                    transform=T.Compose([T.ToTensor(), self.normalizer]),
                    download=True,
                ),
                valid_indices,
            )
            self.test_dataset = Subset(self.test_dataset, test_indices)
        else:
            # If no subset is specified, create validation set from full training set
            num_valid = int(len(full_train_dataset) * self.valid_split)
            valid_indices = list(range(num_valid))
            train_indices = list(range(num_valid, len(full_train_dataset)))

            self.train_dataset = Subset(self.train_dataset, train_indices)
            self.valid_dataset = Subset(
                CIFAR10(
                    root=self.data_dir,
                    train=True,
                    transform=T.Compose([T.ToTensor(), self.normalizer]),
                    download=True,
                ),
                valid_indices,
            )

        print(f"Dataset sizes:")
        print(f"Train: {len(self.train_dataset)}")
        print(f"Validation: {len(self.valid_dataset)}")
        print(f"Test: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
