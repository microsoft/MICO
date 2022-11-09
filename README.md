# MICO
 
Welcome to the Microsoft Membership Inference Competition (MICO)!

In this competition, you will evaluate the effectiveness of differentially private model training as a mitigation against white-box membership inference attacks.

* [What is Membership Inference?](#what-is-membership-inference)
* [What is MICO?](#what-is-mico)
* [Task Details](#task-details)
* [Submissions and Scoring](#submissions-and-scoring)
* [Winner Selection](#winner-selection)
* [Important Dates](#important-dates)
* [Terms and Conditions](#terms-and-conditions)
* [CodaLab Competitions](#codalab-competitions)
* [Getting Started](#getting-started)
* [Contact](#contact)
* [Contributing](#contributing)
* [Trademarks](#trademarks)

## What is Membership Inference?

Membership inference is a widely-studied class of threats against Machine Learning (ML) models.
The goal of a membership inference attack is to infer whether a given record was used to train a specific ML model.
An attacker might have full access to the model and its weights (known as "white-box" access), or might only be able to query the model on inputs of their choice ("black-box" access).
In either case, a successful membership inference attack could have negative consequences, especially if the model was trained on sensitive data.

Membership inference attacks vary in complexity.
In a simple case, the model might have overfitted to its training data, so that it outputs higher confidence predictions when queried on training records than when queried on records that the model has not seen during training.
Recognizing this, an attacker could simply query the model on records of their interest, establish a threshold on the model's confidence, and infer that records with higher confidence are likely members of the training data.
In a white-box setting, as is the case for this competition, the attacker can use more sophisticated strategies that exploit access to the internals of the model.


## What is MICO?

In MICO, your goal is to perform white-box membership inference against a series of trained ML models that we provide.
Specifically, given a model and a set of *challenge points*, the aim is to decide which of these challenge points were used to train the model.

You can compete on any of four separate membership inference tasks against classification models for image, text, and tabular data, as well as on a special _Differential Privacy Distinguisher_ task spanning all 3 modalities.
Each task will be scored separately.
You do not need to participate in all of them, and can choose to participate in as many as you like.
Throughout the competition, submissions will be scored on a subset of the evaluation data and ranked on a live scoreboard.
When submission closes, the final scores will be computed on a separate subset of the evaluation data.

The winner of each task will be eligible for an award of **$2,000 USD** from Microsoft and the runner-up of each task for an award of **$1,000 USD** from Microsoft (in the event of tied entries, these awards may be adjusted).
This competition is co-located with the [IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) 2023](https://satml.org/), and the winners will be invited to present their strategies at the conference.


## Task Details

For each of the four tasks, we provide a set of models trained on different splits of a public dataset.
For each of these models, we provide `m` challenge points; exactly half of which are _members_ (i.e., used to train the model) and half are _non-members_ (i.e., they come from the same dataset, but were not used to train the model).
Your goal is to determine which challenge points are members and which are non-members.

Each of the first three tasks consists of three different _scenarios_ with increasing difficulty, determined by the differential privacy guarantee of the algorithm used to train target models: $\varepsilon = \infty$, high $\varepsilon$, and low $\varepsilon$.
All scenarios share the same model architecture and are trained for the same number of epochs.
The $\varepsilon = \infty$ scenario uses Stochastic Gradient Descent (SGD) without any differential privacy guarantee, while the high $\varepsilon$ and low $\varepsilon$ scenarios use Differentially-Private SGD with a high and low privacy budget $\varepsilon$, respectively.
The lower the privacy budget $\varepsilon$, the more _private_ the model.

In the fourth task, the target models span all three modalities (image, text, and tabular data) and are trained with a low privacy budget.
The model architectures and hyperparameters are the same as for first three tasks.
However, we reveal the training data of models except for the `m/2` member challenge points.


| Task        | Scenario | Dataset      | Model Architecture         | $\varepsilon$ | Other training points given |
| :---           |    :----:   |  :----: |        :----:      |    :----:  |            :----:          |
| Image            | I1       | CIFAR-10     | 4-layer CNN                | $\infty$      | No                           |
|                  | I2       | CIFAR-10     | 4-layer CNN                | High          | No                           |
|                  | I3       | CIFAR-10     | 4-layer CNN                | Low           | No                           |
| Text             | X1       | SST-2        | Roberta-Base               | $\infty$      | No                           |
|                  | X2       | SST-2        | Roberta-Base               | High          | No                           |
|                  | X3       | SST-2        | Roberta-Base               | Low           | No                           |
| Tabular Data     | T1       | Purchase-100 | 3-layer fully connected NN | $\infty$      | No                           |
|                  | T2       | Purchase-100 | 3-layer fully connected NN | High          | No                           |
|                  | T3       | Purchase-100 | 3-layer fully connected NN | Low           | No                           |
| DP Distinguisher | D1       | CIFAR-10     | 4-layer CNN                | Low           | Yes                           |
|                  | D2       | SST-2        | Roberta-Base               | Low           | Yes                           |
|                  | D3       | Purchase-100 | 3-layer fully connected NN | Low           | Yes                           |


## Submissions and Scoring

Submissions will be ranked based on their performance in white-box membership inference against the provided models.

There are three sets of challenges: `train`, `dev`, and `final`.
For models in `train`, we reveal the full training dataset, and consequently the ground truth membership data for challenge points.
These models can be used by participants to develop their attacks.
For models in the `dev` and `final` sets, no ground truth is revealed and participants must submit their membership predictions for challenge points.

During the competition, there will be a live scoreboard based on the `dev` challenges.
The final ranking will be decided on the `final` set; scoring for this dataset will be withheld until the competition ends.

For each challenge point, the submission must provide a value, indicating the confidence level with which the challenge point is a member.
Each value must be a floating point number in the range `[0.0, 1.0]`, where `1.0` indicates certainty that the challenge point is a member, and `0.0` indicates certainty that it is a non-member.

Submissions will be evaluated according to their **True Positive Rate at 10% False Positive Rate** (i.e. `TPR @ 0.1 FPR`).
In this context, *positive* challenge points are members and *negative* challenge points are non-members.
For each submission, the scoring program concatenates the confidence values for all models (`dev` and `final` treated separately) and compares these to the reference ground truth.
The scoring program determines the minimum confidence threshold for membership such that at most 10% of the non-member challenge points are incorrectly classified as members.
The score is the True Positive Rate achieved by this threshold (i.e., the proportion of correctly classified member challenge points).
The live scoreboard shows additional scores (i.e., TPR at other FPRs, membership inference advantage, accuracy, AUC-ROC score), but these are only informational.

You are allowed to make multiple submissions, but only your latest submission will be considered.
In order for a submission to be valid, you must submit confidence values for all challenge points in all three scenarios of the task.

Hints and tips:
- We do realize that the score of a submission leaks some information about the ground truth.
However, using this information to optimize a submission based only on the live scoreboard (i.e., on `dev`) is a bad strategy, as this score has no relevance on the final ranking.
- Pay a special attention to the evaluation metric (`TPR @ 0.1 FPR`).
Your average accuracy at predicting membership in general may be misleading. Your attack should aim to maximize the number of predicted members whilst remaining below the specified FPR.


## Winner Selection

Winners will be selected independently for each task (i.e. if you choose not to participate in certain tasks, this will not affect your rank for the tasks in which you do participate).
For each task, the winner will be the one achieving the highest average score (`TPR @ 0.1 FPR`) across the three scenarios.


## Important Dates

- Submission opens: November 8, 2022
- Submission closes: **January 12, 2023, 23:59 (Anywhere on Earth)**
- Conference: February 8-10, 2023


## Terms and Conditions

- This challenge is subject to the [Microsoft Bounty Terms and Conditions](https://www.microsoft.com/en-us/msrc/bounty-terms).

- Microsoft employees and students/employees of Imperial College London may submit solutions, but are not eligible to receive awards.

- Submissions will be evaluated by a panel of judges according to the aims of the competition.

- Winners may be asked to provide their code and/or a description of their strategy to the judges for verification purposes.

## CodaLab Competitions

- [Image (CIFAR-10)](https://codalab.lisn.upsaclay.fr/competitions/8551)
- [Text (SST-2)](https://codalab.lisn.upsaclay.fr/competitions/8554)
- [Tabular Data (Purchase-100)](https://codalab.lisn.upsaclay.fr/competitions/8553)
- [DP Distinguisher](https://codalab.lisn.upsaclay.fr/competitions/8552)

## Getting Started

First, register on CodaLab for the tasks in which you would like to participate.
Once registered, you will be given URLs from which to download the challenge data.

This repository contains starting kit Jupyter notebooks which will guide you through making your first submission.
To use it, clone this repository and follow the steps below:
- `pip install -r requirements.txt`. You may want to do this in a [virtualenv](https://docs.python.org/3/library/venv.html).
- `pip install -e .`
- `cd starting-kit/`
- `pip install -r requirements-starting-kit.txt`
- The corresponding starting kit notebook illustrates how to load the challenge data, run a basic membership inference attack, and prepare an archive to submit to CodaLab.


## Contact

For any additional queries or suggestions, please contact [mico-competition@microsoft.com](mico-competition@microsoft.com).


## Contributing

This project welcomes contributions and suggestions.
Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment).
Simply follow the instructions provided by the bot.
You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


![Mico Argentatus (Silvery Marmoset) - William Warby/Flickr](codalab-package/logo.png)

Mico Argentatus (Silvery Marmoset) - William Warby/Flickr

