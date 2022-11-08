"""Scoring program for the CodaLab competition platform.

Usage:
    score.py <input directory> <output directory>

This program expects the following directory structure for <input directory>:
- <input directory>/ref/: Contains the solutions directories
    (e.g., cifar10/cifar10_lo, cifar10/cifar10_hi, cifar10/cifar10_inf) for either the dev or the final phase.
- <input directory>/res/: Contains the predictions directories (containing prediction.csv) for both
    phases.
"""
import os
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
from typing import List, Dict

FPR_THRESHOLD = 0.1
FPR_THRESHOLD_LIST = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]

def tpr_at_fpr(true_membership: List, predictions: List, max_fpr=FPR_THRESHOLD) -> float:
    """Calculates the best True Positive Rate when the False Positive Rate is
    at most `max_fpr`.

    Args:
        true_membership (List): A list of values in {0,1} indicating the membership of a
            challenge point. 0: "non-member", 1: "member".
        predictions (List): A list of values in the range [0,1] indicating the confidence
            that a challenge point is a member. The closer the value to 1, the more
            confident the predictor is about the hypothesis that the challenge point is
            a member.
        max_fpr (float, optional): Threshold on the FPR. Defaults to 0.1.

    Returns:
        float: The TPR @ `max_fpr` FPR.
    """
    fpr, tpr, _ = roc_curve(true_membership, predictions)

    return max(tpr[fpr < max_fpr])


def score(solutions: List, predictions: List) -> Dict:
    scores = {}
    for max_fpr in FPR_THRESHOLD_LIST:
        scores[f"TPR_FPR_{int(1e4 * max_fpr)}"] = tpr_at_fpr(solutions, predictions, max_fpr=max_fpr)
    fpr, tpr, _ = roc_curve(solutions, predictions)
    scores["fpr"] = fpr
    scores["tpr"] = tpr
    scores["AUC"] = roc_auc_score(solutions, predictions)
    scores["MIA"] = np.max(tpr - fpr)
    # This is the balanced accuracy, which coincides with accuracy for balanced classes
    scores["accuracy"] = np.max(1 - (fpr + (1 - tpr)) / 2)

    return scores


if __name__ == "__main__":
    from score_html import generate_html

    # Parse arguments.
    assert len(os.sys.argv) == 3, "Usage: score.py <predictions and solutions directory> <output directory>"
    solutions_dir = os.path.join(os.sys.argv[1], "ref")
    predictions_dir = os.path.join(os.sys.argv[1], "res")
    output_dir = os.sys.argv[2]

    current_phase = None

    # Which competition?
    dataset = os.listdir(solutions_dir)
    assert len(dataset) == 1, f"Wrong content: {solutions_dir}: {dataset}"
    dataset = dataset[0]
    print(f"[*] Competition: {dataset}")

    # Update solutions and predictions directories.
    solutions_dir = os.path.join(solutions_dir, dataset)
    assert os.path.exists(solutions_dir), f"Couldn't find soultions directory: {solutions_dir}"

    predictions_dir = os.path.join(predictions_dir, dataset)
    assert os.path.exists(predictions_dir), f"Couldn't find predictions directory: {predictions_dir}"

    scenarios = sorted(os.listdir(solutions_dir))
    assert len(scenarios) == 3, f"Found spurious directories in solutions directory: {solutions_dir}: {scenarios}"

    found_scenarios = sorted(os.listdir(predictions_dir))
    assert scenarios == found_scenarios, f"Found spurious directories in predictions directory {solutions_dir}: {found_scenarios}"

    # Compute the scores for each scenario
    all_scores = {}
    for scenario in scenarios:
        print(f"[*] Processing {scenario}...")

        # What phase are we in?
        phase = os.listdir(os.path.join(solutions_dir, scenario))
        assert len(phase) == 1, "Corrupted solutions directory"
        assert phase[0] in ["dev", "final"], "Corrupted solutions directory"
        current_phase = phase[0]
        print(f"[**] Scoring `{current_phase}` phase...")

        # We compute the scores globally, across the models. This is somewhat equivalent to having
        # one attack (threshold) for all the attacks.
        # Load the predictions.
        predictions = []
        solutions  = []
        for model_id in os.listdir(os.path.join(solutions_dir, scenario, current_phase)):
            basedir = os.path.join(scenario, current_phase, model_id)
            solutions.append(np.loadtxt(os.path.join(solutions_dir, basedir, "solution.csv"), delimiter=","))
            predictions.append(np.loadtxt(os.path.join(predictions_dir, basedir, "prediction.csv"), delimiter=","))

        solutions = np.concatenate(solutions)
        predictions = np.concatenate(predictions)

        # Verify that the predictions are valid.
        assert len(predictions) == len(solutions)
        assert np.all(predictions >= 0), "Some predictions are < 0"
        assert np.all(predictions <= 1), "Some predictions are > 1"

        scores = score(solutions, predictions)

        print(f"[*] Scores: {scores}")
        all_scores[scenario] = scores

    # Store the scores.
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scores.txt"), "w") as f:
        for i, scenario in enumerate(scenarios):
            assert scenario in all_scores, f"Score for scenario {scenario} not found. Corrupted ref/?"
            for score in {"AUC", "MIA", "accuracy"}:
                f.write(f"scenario{i+1}_{score}: {all_scores[scenario][score]}\n")
            for max_fpr in FPR_THRESHOLD_LIST:
                score = f"TPR_FPR_{int(1e4 * max_fpr)}"
                f.write(f"scenario{i+1}_{score}: {all_scores[scenario][score]}\n")

        # Average TPR@0.1FPR (used for ranking)
        avg = np.mean([all_scores[scenario]["TPR_FPR_1000"] for scenario in scenarios])
        f.write(f"average_TPR_FPR_1000: {avg}")

    # Detailed scoring (HTML)
    html = generate_html(all_scores)
    with open(os.path.join(output_dir, "scores.html"), "w") as f:
        f.write(html)
