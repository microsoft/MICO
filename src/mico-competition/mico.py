from __future__ import annotations

import os
import torch
import torch.nn as nn

from collections import OrderedDict
from typing import List, Optional, Union, Type, TypeVar
from torch.utils.data import Dataset, ConcatDataset, random_split

D = TypeVar("D", bound="ChallengeDataset")

LEN_CHALLENGE = 100

class ChallengeDataset:
    """Reconstructs the data splits associated with a model from stored seeds.

    Given a `torch.utils.Dataset`, the desired length of the training dataset `n`,
    and the desired number of members/non-member challenge examples `m`, it uses
    `torch.utils.data.random_split` with the stored seeds to produce:

    - `challenge` : `2m` challenge examples
    - `nonmember` : `m` non-members challenge examples from `challenge`
    - `member`    : `m` member challenge examples, from `challenge`
    - `training`  : non-challenge examples to use for model training
    - `evaluation`: non-challenge examples to use for model evaluation

    Use `get_training_dataset` to construct the full training dataset
    (the concatenation of `member` and `training`) to train a model.

    Use `get_eval_dataset` to retrieve `evaluation`. Importantly, do not
    attempt to use `nonmember` for model evaluation, as releasing the
    evaluation results would leak membership information.

    The diagram below details the process, where arrows denote calls to
    `torch.utils.data.random_split` and `N = len(dataset)`:

          ┌────────────────────────────────────────────────────────────┐
          │                           dataset                          │
          └──────────────────────────────┬─────────────────────────────┘
                                         │N
                          seed_challenge │
                    ┌────────────────────┴────────┐
                    │2m                           │N - 2m
                    ▼                             ▼
          ┌───────────────────┬────────────────────────────────────────┐
          │     challenge     │                 rest                   │
          └─────────┬─────────┴───────────────────┬────────────────────┘
                    │2m                           │N - 2m
    seed_membership │               seed_training │
               ┌────┴────┐              ┌─────────┴────────┐
               │m        │m             │n - m             │N - n - m
               ▼         ▼              ▼                  ▼
          ┌─────────┬─────────┬───────────────────┬────────────────────┐
          │nonmember│ member  │     training      │     evaluation     │
          └─────────┴─────────┴───────────────────┴────────────────────┘

    - Models are trained on `member + training` and evaluated on `evaluation`
    - Standard scenarios disclose `challenge` (equivalently, `seed_challenge`)
    - DP distinguisher scenarios also disclose `training` and `evaluation` (equivalently, `seed_training`)
    - To disclose ground truth, disclose `nonmember` and `member` (equivalently, `seed_membership`)
    """
    def __init__(self, dataset: Dataset, len_training: int, len_challenge: int,
                 seed_challenge: int, seed_training: Optional[int], seed_membership: Optional[int]) -> None:
        """Pseudorandomly select examples for `challenge`, `non-member`, `member`, `training`, and `evaluation`
        splits from given seeds. Only the seed for `challenge` is mandatory.

        Args:
            dataset (Dataset): Dataset to select examples from.
            len_training (int): Length of the training dataset.
            len_challenge (int): Number of challenge examples (`len_challenge` members and `len_challenge` non-members).
            seed_challenge (int): Seed to select challenge examples.
            seed_training (Optional[int]): Seed to select non-challenge training examples.
            seed_membership (Optional[int]): Seed to split challenge examples into members/non-members.
        """
        from torchcsprng import create_mt19937_generator

        challenge_gen = create_mt19937_generator(seed_challenge)
        self.challenge, self.rest = random_split(
            dataset,
            [2 * len_challenge, len(dataset) - 2 * len_challenge],
            generator = challenge_gen)

        if seed_training is not None:
            training_gen = create_mt19937_generator(seed_training)
            self.training, self.evaluation = random_split(
                self.rest,
                [len_training - len_challenge, len(dataset) - len_training - len_challenge],
                generator = training_gen)

        if seed_membership is not None:
            membership_gen = create_mt19937_generator(seed_membership)
            self.nonmember, self.member = random_split(
                self.challenge,
                [len_challenge, len_challenge],
                generator = membership_gen)

    def get_challenges(self) -> Dataset:
        """Returns the challenge dataset.

        Returns:
            Dataset: The challenge examples.
        """
        return self.challenge

    def get_train_dataset(self) -> Dataset:
        """Returns the training dataset.

        Raises:
            ValueError: If the seed to select non-challenge training examples has not been set.
            ValueError: If the seed to split challenges into members/non-members has not been set.

        Returns:
            Dataset: The training dataset.
        """
        if self.training is None:
            raise ValueError("The seed to generate the training dataset has not been set.")

        if self.member is None:
            raise ValueError("The seed to split challenges into members/non-members has not been set.")

        return ConcatDataset([self.member, self.training])

    def get_eval_dataset(self) -> Dataset:
        """Returns the evaluation dataset.

        Raises:
            ValueError: If the seed to generate the evaluation dataset has not been set.

        Returns:
            Dataset: The evaluation dataset.
        """
        if self.evaluation is None:
            raise ValueError("The seed to generate the evaluation dataset has not been set.")

        return self.evaluation

    def get_solutions(self) -> List:
        """Returns the membership labels of the challenges.

        Raises:
            ValueError: If the seed to generate the evaluation dataset has not been set.

        Returns:
            List: The list of membership labels for challenges, indexed as in the
            Dataset returned by `get_challenges()`.
        """
        if self.member is None:
            raise ValueError("The seed to split challenges into members/non-members has not been set.")

        member_indices = set(self.challenge.indices[i] for i in self.member.indices)

        labels = [1 if i in member_indices else 0 for i in self.challenge.indices]

        return labels

    @classmethod
    def from_path(cls: Type[D], path: Union[str, os.PathLike], dataset: Dataset, len_training: int, len_challenge: int=LEN_CHALLENGE) -> D:
        """Loads a ChallengeDataset from a directory `path`.
        The directory must contain, at a minimum, the file `seed_challenge`.

        Args:
            path (str): Path to the folder containing the dataset.

        Returns:
            ChallengeDataset: The loaded ChallengeDataset.
        """
        # Load the seeds.
        if os.path.exists(os.path.join(path, "seed_challenge")):
            with open(os.path.join(path, "seed_challenge"), "r") as f:
                seed_challenge = int(f.read())
        else:
            raise Exception(f"`seed_challenge` was not found in {path}")

        seed_training = None
        if os.path.exists(os.path.join(path, "seed_training")):
            with open(os.path.join(path, "seed_training"), "r") as f:
                seed_training = int(f.read())

        seed_membership = None
        if os.path.exists(os.path.join(path, "seed_membership")):
            with open(os.path.join(path, "seed_membership"), "r") as f:
                seed_membership = int(f.read())

        return cls(
            dataset=dataset,
            len_training=len_training,
            len_challenge=len_challenge,
            seed_challenge=seed_challenge,
            seed_training=seed_training,
            seed_membership=seed_membership
        )


X = TypeVar("X", bound="CNN")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=8, stride=2, padding=3), nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(128, 256, kernel_size=3), nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=6400, out_features=10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x is [B, 3, 32, 32] for CIFAR10
        logits = self.cnn(x)
        return logits

    @classmethod
    def load(cls: Type[X], path: Union[str, os.PathLike]) -> X:
        model = cls()
        state_dict = torch.load(path)
        new_state_dict = OrderedDict((k.replace('_module.', ''), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
        model.eval()
        return model


Y = TypeVar("Y", bound="MLP")

class MLP(nn.Module):
    """
    The fully-connected network architecture from Bao et al. (2022).
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(600, 128), nn.Tanh(),
            nn.Linear(128, 100)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    @classmethod
    def load(cls: Type[Y], path: Union[str, os.PathLike]) -> Y:
        model = cls()
        state_dict = torch.load(path)
        new_state_dict = OrderedDict((k.replace('_module.', ''), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
        model.eval()
        return model


def load_model(task: str, path: Union[str, os.PathLike]) -> nn.Module:
    if task == 'cifar10':
        return CNN.load(os.path.join(path, 'model.pt'))
    elif task == 'purchase100':
        return MLP.load(os.path.join(path, 'model.pt'))
    elif task == 'sst2':
        from transformers import AutoModelForSequenceClassification
        # tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
        model.eval()
        return model
    else:
        raise ValueError("`task` must be one of {'cifar10', 'purchase100', 'sst2'}")
