## Getting Started

The challenge data for this task can be downloaded from: **TODO: update when live**.

The starting kit notebook for this task is available at: **TODO: update when live**.

In the starting kit notebook you will find a walk-through of how to load the data and make your first submission. 
We also provide a library for loading the data with the appropriate splits. This section describes the dataset splits, model training, and answer submission format.


## Challenge Construction

For each dataset and each $\varepsilon$ value, we trained 200 different models.
Each model was trained on a different split of the dataset, which is defined by three seed values: `seed_challenge`, `seed_training`, `seed_membership`.
The diagram below illustrates the splits.
Each arrow denotes a call to `torch.utils.data.random_split` and the labels on the arrows indicate the number of records in each split e.g. `N = len(dataset)`:

```
Parameters:
    - `challenge` : `2m` challenge examples (m = 100)
    - `nonmember` : `m` non-members challenge examples from `challenge`
    - `member`    : `m` member challenge examples, from `challenge`
    - `training`  : non-challenge examples to use for model training
    - `evaluation`: non-challenge examples to use for model evaluation

          ┌────────────────────────────────────────────────────────────┐
          │                           dataset                          │
          └──────────────────────────────┬─────────────────────────────┘
                                         │ N
                          seed_challenge │
                    ┌────────────────────┴────────┐
                    │ 2m                          │ N - 2m
                    ▼                             ▼
          ┌───────────────────┬────────────────────────────────────────┐
          │     challenge     │                 rest                   │
          └─────────┬─────────┴───────────────────┬────────────────────┘
                    │ 2m                          │ N - 2m
    seed_membership │               seed_training │
               ┌────┴────┐              ┌─────────┴────────┐
               │ m       │ m            │ n - m            │ N - n - m
               ▼         ▼              ▼                  ▼
          ┌─────────┬─────────┬───────────────────┬────────────────────┐
          │nonmember│ member  │     training      │     evaluation     │
          └─────────┴─────────┴───────────────────┴────────────────────┘
```

Models are trained on `member + training` and evaluated on `evaluation`.
Standard scenarios disclose `challenge` (equivalently, `seed_challenge`).
DP distinguisher scenarios also disclose `training` and `evaluation` (equivalently, `seed_training`).
The ground truth (i.e., `nonmember` and `member`) can be recovered from `seed_membership`.

The 200 models are split into 3 sets:

- `train` [`model_0` ... `model_99`]: for these models, we provide *full* information (including `seed_membership`). They can be used for training your attack (e.g., shadow models).
- `dev` [`model_100` ... `model_149`]: these models are used for the live scoreboard. Performance on these models has no effect in the final ranking.
- `final` [`model_150` ... `model_199`]: these models are used for deciding the final winners. Attack performance on these models will be only be revealed at the end of the competition.


## Challenge Data

The challenge data provided to participants is arranged as follows:

- `train/`
    - `model_0/`
        - `seed_challenge`: Given this seed, you'll be able to retrieve the challenge points.
        - `seed_training`: Given this seed, you'll be able to retrieve the training points (excluding 50% of the challenge points).
        - `seed_membership`: Given this seed, you'll be able to retrieve the true membership of the challenge points.
        - `model.pt`: The trained model. (Equivalently, `pytorch_model.bin` and `config.json` for text classification models.)
        - `solution.csv`: A list of `{0,1}` values, indicating the true membership of the challenge points.
    - ...
    - `model_99`
        - ...

- `dev/`: Used for live scoring.
    - `model_100`
        - `seed_challenge`
        - `model.pt` (or `pytorch_model.bin` and `config.json`)
    - ...
    - `model_149`
        - ...

- `final/`: Used for final scoring, which will be used to determine the winner.
    - `model_150`:
        - `seed_challenge`
        - `model.pt` (or `pytorch_model.bin` and `config.json`)
    - ...
    - `model_199`:
        - ...

`train` data is provided for your convenience: it contains full information about the membership of the challenge points. 
You can use it for developing your attack (e.g. as shadow models).

You can load the public datasets and individual models and their associated challenge data using the functions provided by the `mico-competition` package in the [accompanying repository](https://github.com/microsoft/MICO) (i.e., `loda_cifar10`, `load_model`, `ChallengeDataset.from_path`, etc.)
Please refer to the starting kit for more information.


## Predictions

You must submit predictions for `dev` and `final` data. 
These will be used for live scoring and final scoring respectively.

Predictions should be provided in **a single `.zip` file** containing the following structure:

- `dev/`: Used for live scoring.
    - `model_100`
        - `predictions.csv`: Provided by the participant. A list of values between 0 and 1, indicating membership confidence for each challenge point. Each value must be a floating point number in the range `[0.0, 1.0]`, where `1.0` indicates certainty that the challenge point is a member, and `0.0` indicates certainty that it is a non-member.
    - `model_101`
        - `predictions.csv`
    - ...
- `final/`: Used for final scoring, which will be used to determine the winners.
    - `model_150`
        - `predictions.csv`: Provided by the participant. A list of confidence values between 0 and 1, indicating membership confidence for each challenge point. Each value must be a floating point number in the range `[0.0, 1.0]`, where `1.0` indicates certainty that the challenge point is a member, and `0.0` indicates certainty that it is a non-member.
    - ...

The starting kit notebooks in the [accompanying repository](https://github.com/microsoft/MICO) provide example code for preparing a submission.

**IMPORTANT: predictions for `dev` and `final` models must be provided for every submission you make.**
