import os
import argparse
import warnings
import git
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchcsprng import create_mt19937_generator, create_random_device_generator

from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

from prv_accountant.dpsgd import find_noise_multiplier

from accountant import PRVAccountant

from mico_competition import ChallengeDataset, CNN, load_cifar10

from tqdm import tqdm, trange

from datetime import datetime

from typing import Callable, Optional


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).mean()


def train(args: argparse.Namespace,
          model: nn.Module,
          device: torch.device,
          train_loader: DataLoader,
          criterion,
          optimizer: optim.Optimizer,
          epoch: int,
          compute_epsilon: Optional[Callable[[int], float]] = None):
    model.train()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=args.max_physical_batch_size,
        optimizer=optimizer
    ) as memory_safe_data_loader:

        if args.disable_dp:
            data_loader = train_loader
        else:
            data_loader = memory_safe_data_loader

        # BatchSplittingSampler.__len__() approximates (badly) the length in physical batches
        # See https://github.com/pytorch/opacus/issues/516
        # We instead heuristically keep track of logical batches processed
        pbar = tqdm(data_loader, desc="Batch", unit="batch", position=1, leave=True, total=len(train_loader), disable=None)
        logical_batch_len = 0
        for i, (inputs, target) in enumerate(data_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            logical_batch_len += len(target)
            if logical_batch_len >= args.batch_size:
                pbar.update(1)
                logical_batch_len = logical_batch_len % args.max_physical_batch_size

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (pbar.n + 1) % args.logging_steps == 0 or (pbar.n + 1) == pbar.total:
                if not args.disable_dp:
                    epsilon = compute_epsilon(delta=args.target_delta)
                    pbar.set_postfix(
                        epoch=f"{epoch:02}",
                        train_loss=f"{np.mean(losses):.3f}",
                        accuracy=f"{np.mean(top1_acc) * 100:.3f}",
                        dp=f"(ε={epsilon:.2f}, δ={args.target_delta})"
                    )
                else:
                    pbar.set_postfix(
                        epoch=f"{epoch:02}",
                        train_loss=f"{np.mean(losses):.3f}",
                        accuracy=f"{np.mean(top1_acc) * 100:.3f}",
                        dp="(ε = ∞, δ = 0)"
                    )

        pbar.update(pbar.total - pbar.n)


def test(args: argparse.Namespace,
         model: nn.Module,
         device: torch.device,
         test_loader: DataLoader,
         criterion):
    model.eval()

    losses = []
    top1_acc = []

    with torch.no_grad():
        for inputs, target in tqdm(test_loader, desc="Test ", unit="batch", disable=None):
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)
    loss_avg = np.mean(losses)

    print(
        f"Test Loss    : {loss_avg:.6f}\n"
        f"Test Accuracy: {top1_avg * 100:.6f}"
    )

    return np.mean(top1_acc)


def main(args: argparse.Namespace):
    noise_generator = None
    if not args.secure_mode and args.train_seed is not None:
        # Following the advice on https://pytorch.org/docs/1.8.1/notes/randomness.html

        if torch.cuda.is_available():
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
            torch.use_deterministic_algorithms(True)
            torch.cuda.manual_seed(args.train_seed)
            torch.cuda.manual_seed_all(args.train_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        import random
        random.seed(args.train_seed)
        os.environ['PYTHONHASHSEED'] = str(args.train_seed)

        # Required to get deterministic batches because Opacus uses secure_rng as a generator for
        # train_loader when poisson_sampling = True even though secure_mode = False, which sets secure_rng = None
        # https://github.com/pytorch/opacus/blob/5e632cdb8d497aade29e5555ad79921c239c78f7/opacus/privacy_engine.py#L206
        torch.manual_seed(args.train_seed)
        np.random.seed(args.train_seed)
        noise_generator = create_mt19937_generator(args.train_seed)

    if (args.seed_challenge is None or args.seed_training is None or args.seed_membership is None):

        if args.split_seed is None:
            seed_generator = create_random_device_generator()
        else:
            seed_generator = create_mt19937_generator(args.split_seed)

        args.seed_challenge, args.seed_training, args.seed_membership = torch.empty(
            3, dtype=torch.int64).random_(0, to=None, generator=seed_generator)

        print("Using generated seeds\n"
             f"  seed_challenge  = {args.seed_challenge}\n"
             f"  seed_training   = {args.seed_training}\n"
             f"  seed_membership = {args.seed_membership}\n")

    else:
        print("Using specified seeds")

    full_dataset = load_cifar10(dataset_dir=args.dataset_dir, download=False)

    challenge_dataset = ChallengeDataset(
        full_dataset,
        len_challenge=args.len_challenge,
        len_training=args.len_training,
        seed_challenge=args.seed_challenge,
        seed_training=args.seed_training,
        seed_membership=args.seed_membership)

    train_dataset = challenge_dataset.get_train_dataset()
    test_dataset  = challenge_dataset.get_eval_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.max_physical_batch_size,
        num_workers=args.dataloader_num_workers
    )

    # Supress warnings
    warnings.filterwarnings(action="ignore", module="opacus", message=".*Secure RNG turned off")
    warnings.filterwarnings(action="ignore", module="torch", message=".*Using a non-full backward hook")

    model = CNN()
    assert ModuleValidator.is_valid(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0)
    
    # Not the same as args.batch_size / len(train_dataset)
    args.sample_rate = 1 / len(train_loader)
    num_steps = int(len(train_loader) * args.num_epochs)

    if not args.disable_dp:
        args.noise_multiplier = find_noise_multiplier(
            sampling_probability=args.sample_rate,
            num_steps=num_steps,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            eps_error=0.1
        )

        privacy_engine = PrivacyEngine(secure_mode=args.secure_mode)

        # Override Opacus accountant
        # Revise if https://github.com/pytorch/opacus/pull/493 is merged
        privacy_engine.accountant = PRVAccountant(
            noise_multiplier=args.noise_multiplier,
            sample_rate=args.sample_rate,
            max_steps=num_steps,
            eps_error=0.1,
            delta_error=1e-9)

        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            poisson_sampling=True,
            noise_generator=noise_generator
        )

        print(f"Training using DP-SGD with {optimizer.original_optimizer.__class__.__name__} optimizer\n"
             f"  noise multiplier σ = {optimizer.noise_multiplier},\n"
             f"  clipping norm C = {optimizer.max_grad_norm:},\n"
             f"  average batch size L = {args.batch_size},\n"
             f"  sample rate = {args.sample_rate},\n"
             f"  for {args.num_epochs} epochs ({num_steps} steps)\n"
             f"  to target ε = {args.target_epsilon}, δ = {args.target_delta}")

        compute_epsilon: Optional[Callable[[float], float]] = lambda delta: privacy_engine.get_epsilon(delta=delta)
    else:
        print(f"Training using SGD with {optimizer.__class__.__name__} optimizer\n"
             f"  batch size L = {args.batch_size},\n"
             f"  for {args.num_epochs} epochs ({num_steps} steps)")
        compute_epsilon = None

    # Must be initialized after attaching the privacy engine.
    # See https://discuss.pytorch.org/t/how-to-use-lr-scheduler-in-opacus/111718
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma)

    pbar = trange(args.num_epochs, desc="Epoch", unit="epoch", position=0, leave=True, disable=None)
    for epoch in pbar:
        pbar.set_postfix(lr=f"{scheduler.get_last_lr()}")
        train(args, model, device, train_loader, criterion, optimizer, epoch + 1, compute_epsilon=compute_epsilon)
        scheduler.step()

    acc = test(args, model, device, test_loader, criterion)
    with open(os.path.join(args.output_dir, "accuracy"), "w") as f:
        print(f"{acc:.3f}", file=f)

    if not args.disable_dp:
        final_epsilon = compute_epsilon(args.target_delta)
        print(f"The trained model is (ε = {final_epsilon}, δ = {args.target_delta})-DP")
        with open(os.path.join(args.output_dir, "epsilon"), "w") as f:
            print(f"{final_epsilon:.3f}", file=f)

    with open(os.path.join(args.output_dir, "seed_challenge"), "w") as f:
        print(f"{args.seed_challenge}", file=f)

    with open(os.path.join(args.output_dir, "seed_training"), "w") as f:
        print(f"{args.seed_training}", file=f)

    with open(os.path.join(args.output_dir, "seed_membership"), "w") as f:
        print(f"{args.seed_membership}", file=f)

    with open(os.path.join(args.output_dir, "solution.csv"), "w") as f:
        solution = challenge_dataset.get_solutions()
        csv.writer(f).writerow(solution)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, metavar='ID',
                        help="an identifier for the trained model")
    # Seeds
    parser.add_argument("--train_seed", type=int, metavar='TS',
                        help="seed for reproducibility")
    parser.add_argument("--split_seed", type=int, metavar='SS',
                        help="seed to deterministically generate the 3 seeds for creating splits "
                        "(--seed_challenge, --seed_trainig, seed_membership)")
    parser.add_argument("--seed_challenge", type=int, metavar='SC',
                        help="seed to select challenge examples")
    parser.add_argument("--seed_training", type=int, metavar='ST',
                        help="seed to select non-challenge training examples")
    parser.add_argument("--seed_membership", type=int, metavar='SM',
                        help="seed to split challenge examples into members/non-members")
    # Split lengths
    parser.add_argument("--len_training", type=int, metavar="N", required=True,
                        help="(required) number of examples used for training")
    parser.add_argument("--len_challenge", type=int, metavar="m", required=True,
                        help="(required) number of member and non-member challenge examples "
                        "(i.e., m members and m non-members)")
    # General
    parser.add_argument("--secure_mode", action="store_true", default=False,
                        help="whether to use Opacus secure mode for training (default=True)")
    parser.add_argument("--disable_dp", action="store_true", default=False,
                        help="whether to disable differentially private training altogether (default=False)")
    parser.add_argument("--dataloader_num_workers", type=int, metavar='W', default=2,
                        help="number of workers for data loading. 0 means that the data will be loaded in the main process (default=2). "
                        "See torch.utils.data.DataLoader")
    parser.add_argument("--logging_steps", type=int, metavar='k', default=10,
                        help="prints accuracy, loss, and privacy accounting information during training every k logical batches "
                        "(default=10)")
    parser.add_argument("--dataset_dir", type=str, metavar="DATA", default=".",
                        help="root directory for cached dataset (default='.')")
    parser.add_argument("--output_dir", type=str, metavar="OUT",
                        help="output directory. If none given, will pick one based on hyperparameters")
    # Training hyperparameters
    parser.add_argument("--target_epsilon", type=float, metavar="EPSILON",
                        help="target DP epsilon. Required unless specifying --disable_dp")
    parser.add_argument("--target_delta", type=float, metavar="DELTA",
                        help="target DP delta. Will use 1/N if unspecified")
    parser.add_argument("--batch_size", type=int, metavar="L",
                        help="expected logical batch size; determines the sample rate of DP-SGD. "
                        "Actual batch size varies because batches are constructed using Poisson sampling")
    parser.add_argument("--max_physical_batch_size", type=int, metavar="B",
                        help="maximum physical batch size. Use to simulate logical batches larger than available memory and "
                        "to safeguard against unusually large batches produces by Poisson sampling. "
                        "See opacus.utils.batch_memory_manager.BatchMemoryManager")
    parser.add_argument("--num_epochs", metavar='E', type=int, default=10,
                        help="number of training epochs (default=10)")
    parser.add_argument("--max_grad_norm", type=float, metavar='C', default=1.0,
                        help="clipping norm for per-sample gradients in DP-SGD (default=1.0)")
    parser.add_argument("--learning_rate", type=float, metavar="LR", default=1.0,
                        help="initial learning rate (default=1.0)")
    parser.add_argument("--lr_scheduler_gamma", type=float, metavar="GAMMA", default=1.0,
                        help="gamma parameter for exponential learning rate scheduler")
    parser.add_argument("--lr_scheduler_step", type=int, metavar="S", default=1,
                        help="step size for exponential learning rate scheduler")

    args = parser.parse_args()

    if args.len_training is None:
        raise ValueError("Please specify --len_training")

    if args.len_challenge is None:
        raise ValueError("Please specify --len_challenge")

    # Parameter validation
    if args.secure_mode and args.train_seed is not None:
        raise ValueError("Specify either secure mode or a seed for reproducibility, but not both")

    if args.target_delta is None:
        args.target_delta = 1 / args.len_training

    if args.split_seed is not None and (args.seed_challenge is not None or args.seed_training is not None or args.seed_membership is not None):
        raise ValueError("A --split_seed was given to generate seeds to construct splits but at least one explicit seed was specified. Bailing out.")

    if args.output_dir is None:
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.disable_dp:
            args.output_dir = f"{now}-nodp-lr{args.learning_rate}-gamma{args.lr_scheduler_gamma}-S{args.lr_scheduler_step}-L{args.batch_size}-" + \
                f"E{args.num_epochs}"
        else:
            args.output_dir = f"{now}-eps{args.target_epsilon}-delta{args.target_delta}-lr{args.learning_rate}-" + \
                f"gamma{args.lr_scheduler_gamma}-S{args.lr_scheduler_step}-L{args.batch_size}-E{args.num_epochs}-C{args.max_grad_norm}" + \
                f"{'-secure' if args.secure_mode else ''}"

        print(f"No --output_dir specified. Will use {args.output_dir}")

    if args.model_id is not None:
        args.output_dir = args.output_dir + f"_{args.model_id}"

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "arguments"), "w") as argfile:
        try:
            commit_hash = git.Repo(".", search_parent_directories=True).git.rev_parse("HEAD")
        except git.exc.InvalidGitRepositoryError:
            commit_hash = "unknown"
        print(f"Commit hash: {commit_hash}")
        print(f"# Commit hash: {commit_hash}", file=argfile)
        for arg in vars(args):
            print(f"--{arg} {getattr(args, arg)}")
            print(f"--{arg} {getattr(args, arg)}", file=argfile)

    main(args)
