---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: index
---

# Announcing the Winners of the MICO Competition

The Microsoft Membership Inference Competition (MICO) was one of the 3 competitions accepted at the [1st IEEE Conference on Secure and Trustworthy Machine Learning (SaTML 2023)](https://satml.org/).
The competition was [launched](https://msrc-blog.microsoft.com/2022/11/16/mico/) in November 2022, and final submissions were due on January 26th, 2023. Today we are excited to announce the winning submissions!

The goal of this competition was to benchmark membership inference attacks against machine learning (ML) models in a variety of settings. ML models are often trained on sensitive datasets, which raises privacy concerns. In a membership inference attack, the adversary aims to infer whether a specific data record was used to train an ML model. Our competition focussed on a white-box threat model, where the adversary has access to the learned weights, model architecture, and training hyperparameters. This setup allows the adversary to train similar _shadow_ models and extract features from them to train an attacker model. Given a model and a set of challenge points, the participants’ task was to decide which of these challenge points were used to train the model, i.e., were members.

## Competition tracks

The competition consisted of four tracks: three tracks using classification models for images (a 4-layer CNN trained on CIFAR-10), tabular data (a 3-layer fully-connected NN trained on Purchase-100), and text (a pre-trained large language model fine-tuned on SST-2), and a fourth track spanning all 3 modalities where participants were given additional information. Each of the first three tracks was subdivided in 3 scenarios of increasing difficulty: models trained without any privacy guarantee and models trained with differential privacy, with _high_ (ε = 10) and low (ε = 4) privacy budgets. For each scenario, we provided participants with 200 models

-	100 _train_ models that participants could use as shadow models, for which we revealed the full training dataset,
-	50 _dev_ models, each with 200 challenge points, which we used to rank submissions during the competition in a live leader board. 
-	50 _test_ models, each with 200 challenge points, which we used to determine the final ranking.

The fourth track was similarly structured in 3 scenarios, one for each modality (images, tabular data, text), with models trained with a low differential privacy budget. However, for _dev_ and _test_ models we revealed the entire training dataset apart from any challenge points.

We hosted the competition tracks as separate competitions in CodaLab:

- [Image (CIFAR-10)](https://codalab.lisn.upsaclay.fr/competitions/8551)
- [Text (SST-2)](https://codalab.lisn.upsaclay.fr/competitions/8554)
- [Tabular Data (Purchase-100)](https://codalab.lisn.upsaclay.fr/competitions/8553)
- [DP Distinguisher](https://codalab.lisn.upsaclay.fr/competitions/8552)


## Scoring and ranking

Submissions were scored and ranked based on their true positive rate at a 10% false positive rate (TPR @ 10% FPR). This represents a realistic attack in which an adversary aims to confidently identify as many members as possible while tolerating only few errors. We plotted full Receiver Operating Characteristic (ROC) curves for attacks, and reported as well their Area Under the Curve (AUC) score, accuracy, and membership inference advantage (TPR - FPR).


## Accessibility

We designed the competition so that participating did not require a large amount of computational resources. In particular, we calibrated the number of shadow models we provided so that participants would not need to train additional models on their own. As a matter of fact, none of the winners trained additional shadow models. To give an idea of how expensive this could be, we used around 600 GPU hours to train the 2,400 models we provided for the competition. Participants were free to enter any subset of the 4 tracks of the competition. The CIFAR-10 and Purchase-100 tracks offered a low bar for entry for interested participants with no access to GPU accelerators, while the SST-2 and DP Distinguisher tracks required larger downloads (256 GB and 88 GB, respectively) and it would have been slower to compute predictions from the corresponding models without access to at least a low-end GPU.  


## Transparency

A more detailed technical description of the competition as well as the code used to train models and score submissions is available in [the competition GitHub repository](https://github.com/microsoft/MICO).

## Results

We received a total of 509 entries from 58 distinct participants across the 4 tracks, with the CIFAR-10 track receiving by far the largest number of entries. 


Track	         | Entries  | Registered participants |
:----------------|---------:|------------------------:|
CIFAR-10         |	300	    | 45                      |
Purchase-100     |	167	    | 29                      |
SST-2	         |  25	    | 29                      |
DP Distinguisher |	17	    | 23                      |

<br />

We congratulate all participants for taking part in this competition, and we are particularly excited to announce the winner and runner-up in each track.


Track	         |Winner                 |	Runner-up
:----------------|:----------------------|:---------------------
CIFAR-10         | tianjian10 [[solution]](https://github.com/tianjian10/MICO/tree/main/src2) | Anshuman Suri (iamgroot42) [[solution]](https://www.anshumansuri.me/post/mico/)
Purchase-100     | tianjian10 [[solution]](https://github.com/tianjian10/MICO/tree/main/src2) | Sarthak Gupta and Pranjal Gulati (mrsarthakgupta) [[solution]](https://github.com/DevPranjal/mico-first-principles)
SST-2            | tianjian10 [[solution]](https://github.com/tianjian10/MICO/tree/main/src2) | Major Will Fleshman (flesh) [[solution]](https://github.com/wfleshman/MICO_SST)
DP Distinguisher | tianjian10 [[solution]](https://github.com/tianjian10/MICO/tree/main/src2) | [deserted]

<br />

The winner of each track is eligible for an award from Microsoft of **$2,000 USD**; runners-up are eligible for an award of **$1,000 USD**.

## Winners

Since we were unable to attend and present in-person at IEEE SaTML 2023, we asked winning participants to publish short write-ups of their approaches, to which we have linked above. We also invited them to provide a few details about themselves, which all of our runners-up did:

### Sarthak Gupta and Pranjal Gulati (runners-up in the Purchase-100 track) 
are 3rd year undergraduate students in the Department of Mathematics and the Department of Electronics and Communications at the Indian Institute of Technology Roorkee.

### Anshuman Suri (runner-up in the CIFAR-10 track) 
is a fourth-year PhD candidate at the University of Virginia, working in the SRG lab with Professor David Evans. His research spans distribution/property inference, and overall notions of privacy and security in machine learning.

### Major Will Fleshman (runner-up in the SST-2 track) 
is a Senior Cyberspace Capabilities Developer in the U.S. Army specializing in data science and machine learning-based capability development. He holds undergraduate degrees in mathematics and computer science and a master's degree in computer science from the Georgia Institute of Technology. Will is set to begin a Ph.D. program this fall and plans on continuing to contribute to academic challenges in his spare time.

### tianjian10 (winner in all tracks) 
The participant chose to remain anonymous.


## Analysis of results. 

As expected, models trained with low DP privacy budget were the most difficult to attack, while those trained without DP guarantees were the most vulnerable. 

Most approaches, including the winning submissions in all tracks were black-box, using only the vector of class probabilities output by models, and did not exploit white-box access to model weights. The winner in all tracks used the provided shadow models to compute membership scores for challenge points, adapting the state-of-the-art [LiRA attack](https://arxiv.org/abs/2112.03570). Notably, the same attack was used for the DP Distinguisher track, without utilizing the additional information available about training points. The runner-up submission in the Purchase-100 track was unique in experimenting with gradient-based features computed using white-box access to models. 

We believe there is still much room for improving attacks, e.g., by leveraging white-box features such as weights and activations of internal layers, training additional shadow models, employing per-class or per-challenge strategies such as leave-one-out attacks, or by leveraging the additional information in the DP distinguisher track.


## Next steps

We plan to publish a technical report about the competition, describing the rationale for the design and an in-depth analysis of winning entries. We also plan to publish the competition data as a benchmark for membership inference attacks and maintain a public leader board tracking state-of-the-art attacks.

