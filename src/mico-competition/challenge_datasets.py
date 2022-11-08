import os
import numpy as np
import torch

from torch.utils.data import Dataset, ConcatDataset


def load_cifar10(dataset_dir: str = ".", download=True) -> Dataset:
    """Loads the CIFAR10 dataset.
    """
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    # Precomputed statistics of CIFAR10 dataset
    # Exact values are assumed to be known, but can be estimated with a modest privacy budget
    # Opacus wrongly uses CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    # This is the _average_ std across all images (see https://github.com/kuangliu/pytorch-cifar/issues/8)
    CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
    CIFAR10_STD  = (0.24703223, 0.24348513, 0.26158784)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # NB: torchvision checks the integrity of downloaded files
    train_dataset = CIFAR10(
        root=f"{dataset_dir}/cifar10",
        train=True,
        download=download,
        transform=transform
    )

    test_dataset = CIFAR10(
        root=f"{dataset_dir}/cifar10",
        train=False,
        download=download,
        transform=transform
    )

    return ConcatDataset([train_dataset, test_dataset])


def load_sst2() -> Dataset:
    """Loads the SST2 dataset.
    """
    import datasets

    # Specify cache_dir as argument?
    ds = datasets.load_dataset("glue", "sst2")
    return ConcatDataset([ds['train'], ds['validation']])


class Purchase100(Dataset):
    """
    Purchase100 dataset pre-processed by Shokri et al.
    (https://github.com/privacytrustlab/datasets/blob/master/dataset_purchase.tgz).
    We save the dataset in a .pickle version because it is much faster to load
    than the original file.
    """
    def __init__(self, dataset_dir: str) -> None:
        import pickle

        dataset_path = os.path.join(dataset_dir, 'purchase100', 'dataset_purchase')

        # Saving the dataset in pickle format because it is quicker to load.
        dataset_path_pickle = dataset_path + '.pickle'

        if not os.path.exists(dataset_path) and not os.path.exists(dataset_path_pickle):
            raise ValueError("Purchase-100 dataset not found.\n"
                             "You may download the dataset from https://www.comp.nus.edu.sg/~reza/files/datasets.html\n"
                            f"and unzip it in the {dataset_dir}/purchase100 directory")

        if not os.path.exists(dataset_path_pickle):
            print('Found the dataset. Saving it in a pickle file that takes less time to load...')
            purchase = np.loadtxt(dataset_path, dtype=int, delimiter=',')
            with open(dataset_path_pickle, 'wb') as f:
                pickle.dump({'dataset': purchase}, f)

        with open(dataset_path_pickle, 'rb') as f:
            dataset = pickle.load(f)['dataset']

        self.labels = list(dataset[:, 0] - 1)
        self.records = torch.FloatTensor(dataset[:, 1:])
        assert len(self.labels) == len(self.records), f'ERROR: {len(self.labels)} and {len(self.records)}'
        print('Successfully loaded the Purchase-100 dataset consisting of',
            f'{len(self.records)} records and {len(self.records[0])}', 'attributes.')

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx], self.labels[idx]


def load_purchase100(dataset_dir: str = ".") -> Dataset:
    """Loads the Purchase-100 dataset.
    """
    return Purchase100(dataset_dir)
