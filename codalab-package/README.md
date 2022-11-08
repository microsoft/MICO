This is the package containing the CodaLab competition template.

# Create a Competition Bundle

Let `$COMPETITION_DATA` be the directory containing all the private competition data directories including the ground truth `solution.csv` files, i.e.: 

```
`$COMPETITION_DATA`
├── {scenario_1}
│   ├── dev
│   ├── final
│   └── train
├── {scenario_2}
│   ├── dev
│   ├── final
│   └── train
└── {scenario_3}
    ├── dev
    ├── final
    └── train
```

To create the competition bundle, from the root directory of this repository:

```
bash codalab-package/utilities/make_competition_bundle.sh $COMPETITION_DATA codalab-package src/mico-competition/scoring
```
The second argument is a path to the codalab package (i.e., this directory).
The third argument is the path to the scoring program.


## Editing the Web Pages

Edit the `.md` files. These are converted to HTML by the competition bundling script.


## CodaLab Internals

On CodaLab's side:

- The `.zip` file containing the predictions of a submission is expanded in `input/res/` on CodaLab's docker.

- The `reference_data_{phase}/*` files for the appropriate challenge `{phase}` are expanded in `input/ref` on CodaLab's docker.

- CodaLab runs `scoring_program/score.py`, which takes as input the `input/` folder, and returns a list of scores into the `output/` folder. Results are stored in `scores.txt` as `score-name: score` lines.
This file is parsed by CodaLab to populate the leaderboards. 
The usage of the scoring program is specified in `scoring_program/metadata`.

- The leaderboards are specified in the `competition.yaml` file together with other configuration parameters of the competition.

When the phase changes from `dev` to `final`, CodaLab will automatically update the predictions with the scores for the `final` challenges.
