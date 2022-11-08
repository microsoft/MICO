# MICO

Welcome to the Microsoft Membership Inference Competition (MICO)! 
In this competition, you will evaluate the effectiveness of differentially private model training as a mitigation against white-box membership inference attacks.


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


## Getting started

Please select the "Participate" tab above, and register for the competition.
Once registered, you will be given URLs from which to download the challenge data.

The [accompanying repository](https://github.com/microsoft/MICO) contains starting kit Jupyter notebooks which will guide you through making your first submission.
To use it, clone the repository and follow the steps below:

- `pip install -r requirements.txt`. You may want to do this in a [virtualenv](https://docs.python.org/3/library/venv.html).
- `pip install -e .`
- `cd starting-kit/`
- `pip install -r requirements-starting-kit.txt`
- The corresponding starting kit notebook illustrates how to load the challenge data, run a basic membership inference attack, and prepare an archive to submit to CodaLab.
