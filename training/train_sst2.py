import numpy as np
import pandas as pd
import os
import torch
import sys
import csv
import yaml
import warnings
import datasets

from opacus import PrivacyEngine

from dp_transformers import TrainingArguments, PrivacyArguments, PrivacyEngineCallback

from prv_accountant.dpsgd import find_noise_multiplier, DPSGDAccountant

from torchcsprng import create_mt19937_generator, create_random_device_generator

from transformers import (
    HfArgumentParser, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, EvalPrediction, PreTrainedTokenizerBase
)

from dataclasses import dataclass
from pathlib import Path

from mico_competition import ChallengeDataset, load_sst2

from typing import Optional


@dataclass
class ModelArguments:
    model_name: str


@dataclass
class DataArguments:
    model_index: int
    len_training: int = 67349
    len_challenge: int = 100
    seed_challenge: Optional[int] = None
    seed_training: Optional[int] = None
    seed_membership: Optional[int] = None
    split_seed: Optional[int] = None


@dataclass
class SecurePrivacyArguments(PrivacyArguments):
    delta: float = None
    use_secure_prng: bool = False


@dataclass
class Arguments:
    training: TrainingArguments
    model: ModelArguments
    privacy: SecurePrivacyArguments
    data: DataArguments


def preprocess_text(D: datasets.DatasetDict, tokenizer: PreTrainedTokenizerBase,
                    max_sequence_length: int = None) -> datasets.DatasetDict:
    processed_data = D.map(
        lambda batch: tokenizer(batch["sentence"], padding="max_length", max_length=max_sequence_length),
        batched=True
    )
    return processed_data.remove_columns(["sentence"])

def load_dataset() -> datasets.DatasetDict:
    if (args.data.seed_challenge is None or args.data.seed_training is None or args.data.seed_membership is None):
        if args.data.split_seed is None:
            seed_generator = create_random_device_generator()
        else:
            seed_generator = create_mt19937_generator(args.split_seed)

        args.data.seed_challenge, args.data.seed_training, args.data.seed_membership = torch.empty(
            3, dtype=torch.int64).random_(0, to=None, generator=seed_generator)

        print("Using generated seeds\n"
                f"  seed_challenge  = {args.data.seed_challenge}\n"
                f"  seed_training   = {args.data.seed_training}\n"
                f"  seed_membership = {args.data.seed_membership}\n")
    else:
        print("Using specified seeds")

    full_dataset = load_sst2()

    challenge_dataset = ChallengeDataset(
        full_dataset,
        len_challenge=args.data.len_challenge,
        len_training=args.data.len_training,
        seed_challenge=args.data.seed_challenge,
        seed_training=args.data.seed_training,
        seed_membership=args.data.seed_membership)

    with open(os.path.join(args.training.output_dir, "challenge", "seed_challenge"), "w") as f:
        print(f"{args.data.seed_challenge}", file=f)

    with open(os.path.join(args.training.output_dir, "challenge", "seed_training"), "w") as f:
        print(f"{args.data.seed_training}", file=f)

    with open(os.path.join(args.training.output_dir, "challenge", "seed_membership"), "w") as f:
        print(f"{args.data.seed_membership}", file=f)

    with open(os.path.join(args.training.output_dir, "challenge", "solution.csv"), "w") as f:
        solution = challenge_dataset.get_solutions()
        csv.writer(f).writerow(solution)

    ds_train = pd.DataFrame.from_records(challenge_dataset.get_train_dataset())
    ds_test = pd.DataFrame.from_records(challenge_dataset.get_eval_dataset())

    return datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(ds_train),
        "test": datasets.Dataset.from_pandas(ds_test)
    }).remove_columns("idx")


def main(args: Arguments):
    output_dir = Path(args.training.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.training.output_dir, "arguments.yml"), "w") as f:
        yaml.dump(args, f)
    print(yaml.dump(args))

    os.mkdir(output_dir/"challenge")

    ds = load_dataset()

    if args.privacy.use_secure_prng:
        import torchcsprng as csprng
        mt19937_gen = csprng.create_mt19937_generator()
        ds['train'] = ds['train'].select(torch.randperm(len(ds['train']), generator=mt19937_gen).tolist())

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    warnings.filterwarnings(action="ignore", module="torch", message=".*Using a non-full backward hook")

    model = AutoModelForSequenceClassification.from_pretrained(args.model.model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name)

    ds = preprocess_text(ds, tokenizer=tokenizer, max_sequence_length=67)

    model.train()
    model = model.to(args.training.device)

    if (not args.training.no_cuda) and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA is not available. Please use --no-cuda to run this script.")

    callbacks = []
    if not args.privacy.disable_dp:
        sampling_probability = training_args.train_batch_size * training_args.gradient_accumulation_steps / len(ds["train"])
        num_steps = int(np.ceil(1 / sampling_probability) * training_args.num_train_epochs)
        noise_multiplier = find_noise_multiplier(
            sampling_probability=sampling_probability, num_steps=num_steps, target_epsilon=args.privacy.target_epsilon,
            target_delta=args.privacy.delta,
            eps_error=0.1
        )
        engine = PrivacyEngine(
            module=model,
            batch_size=training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps,
            sample_size=len(ds['train']),
            noise_multiplier=noise_multiplier,
            max_grad_norm=args.privacy.per_sample_max_grad_norm,
            secure_rng=args.privacy.use_secure_prng,
        )
        accountant = DPSGDAccountant(
            noise_multiplier=noise_multiplier, sampling_probability=sampling_probability, max_steps=num_steps,
            eps_error=0.2
        )
        privacy_callback = PrivacyEngineCallback(
            engine,
            compute_epsilon=lambda s: accountant.compute_epsilon(num_steps=s, delta=args.privacy.delta)[2]
        )
        callbacks.append(privacy_callback)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    trainer = Trainer(
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    try:
        trainer.train()
    finally:
        trainer.save_model(output_dir/"challenge")

    if args.privacy.disable_dp:
        epsilon_final = float('inf')
    else:
        epsilon_final = accountant.compute_epsilon(num_steps=engine.steps, delta=args.privacy.delta)[2]
        trainer.log({"epsilon_final": epsilon_final})
        assert np.isclose(epsilon_final, args.privacy.target_epsilon, atol=0.2, rtol=0.0)

    print("Training successful. Exiting...")
    return 0


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ModelArguments, SecurePrivacyArguments, DataArguments))
    training_args, model_args, privacy_args, data_args = parser.parse_args_into_dataclasses()
    args = Arguments(training=training_args, model=model_args, privacy=privacy_args, data=data_args)
    sys.exit(main(args))
