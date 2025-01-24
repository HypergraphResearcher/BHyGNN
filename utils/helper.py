import random, os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import yaml
import os.path as osp
from itertools import product
from .dataLoader import load_data
from .preprocess import preprocess_data, augmentation


def load_yaml(args):

    file_dir = args.root_dir
    dname = args.dname
    file_path = osp.join(file_dir, "config.yaml")
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    existing_dataset = [
        "20newsW100",
        "ModelNet40",
        "zoo",
        "NTU2012",
        "Mushroom",
        "coauthor_cora",
        "coauthor_dblp",
        "yelp",
        "amazon-reviews",
        "walmart-trips",
        "house-committees",
        "walmart-trips-100",
        "house-committees-100",
        "cora",
        "citeseer",
        "pubmed",
        "twitter",
        "congress-bills",
        "congress-bills-100",
        "senate-committees",
        "senate-committees-100",
    ]

    synthetic_list = [
        "amazon-reviews",
        "walmart-trips",
        "house-committees",
        "walmart-trips-100",
        "house-committees-100",
        "congress-bills",
        "congress-bills-100",
        "senate-committees",
        "senate-committees-100",
    ]

    if dname in existing_dataset:
        if dname in synthetic_list:
            setting = config[f"{dname}-{args.feature_noise:.1f}"]
        else:
            setting = config[dname]
    else:
        raise ValueError("Unknown dataset")
    return setting


def set_parameters(args):
    hyperparameters = load_yaml(args.root_dir)
    if args.dname in hyperparameters.keys():
        parameters = hyperparameters[args.dname]
        args.lr = parameters["lr"]
        args.wd = parameters["wd"]
        args.dropout = parameters["dropout"]
        args.hidden_dim = parameters["hidden_dim"]
        args.MLP_hidden = parameters["MLP_hidden"]
    else:
        raise ValueError("The dataset does not have stored parameters")
    return args


def cosine_similarity(X, rerange=True):
    X = F.normalize(X, dim=1)
    sim = torch.mm(X, X.t())
    return (sim + 1) / 2 if rerange else sim


def euclidean_distance(X):
    #     differences = X.unsqueeze(0) - X.unsqueeze(1)
    #     return torch.sqrt(differences.pow(2).sum(-1))
    sum_X = torch.sum(X * X, dim=1)
    dist = sum_X.unsqueeze(1) + sum_X.unsqueeze(0) - 2 * X @ X.T
    # Ensure numerical stability and take the square root
    dist = torch.sqrt(torch.clamp(dist, min=0))
    return dist


def fix_seed(seed=37):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def evaluate(y_pred, y_true, split_idx, eval_func):
    train_acc = eval_func(y_true[split_idx["train"]], y_pred[split_idx["train"]])
    valid_acc = eval_func(y_true[split_idx["valid"]], y_pred[split_idx["valid"]])
    test_acc = eval_func(y_true[split_idx["test"]], y_pred[split_idx["test"]])

    # Also keep track of losses
    train_loss = F.nll_loss(
        y_pred[split_idx["train"]], y_true[split_idx["train"]]
    ).item()
    valid_loss = F.nll_loss(
        y_pred[split_idx["valid"]], y_true[split_idx["valid"]]
    ).item()
    test_loss = F.nll_loss(y_pred[split_idx["test"]], y_true[split_idx["test"]]).item()

    return (
        train_acc,
        valid_acc,
        test_acc,
        train_loss,
        valid_loss,
        test_loss,
        y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy(),
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger(object):
    """Adapted from https://github.com/snap-stanford/ogb/"""

    def __init__(self, runs, y_true, split_idx_lst):
        self.num_runs = runs
        self.y_true = y_true
        self.split_idx_lst = split_idx_lst
        self.num_nodes = y_true.size
        self.y_preds = [[] for _ in range(self.num_runs)]
        self.results = [[] for _ in range(self.num_runs)]

    def add_result(self, run, result):
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result[:6])
        self.y_preds[run].append(result[6])

    def print_statistics(self, run=None, print_result=True):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            if print_result:
                print(f"\nRun {run + 1:02d}:")
                print(f"Highest Train: {result[:, 0].max():.2f}")
                print(f"Highest Valid: {result[:, 1].max():.2f}")
                print(f"  Final Train: {result[argmax, 0]:.2f}")
                print(f"   Final Test: {result[argmax, 2]:.2f}")
            return argmax, result[argmax, :]
        else:
            result = torch.tensor(self.results)

            best_results = []
            best_epoch = []
            for r in result:
                index = np.argmax(r[:, 1]).item()
                best_epoch.append(index)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f"All runs:")
            print("best epoch:", best_epoch)
            print("Best Test:", np.round(best_result[:, 3].numpy(), 2))
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")

            return best_epoch, best_result

    def plot_result(self, run=None):
        plt.style.use("seaborn")
        if run is not None:
            result = torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f"Run {run + 1:02d}:")
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(["Train", "Valid", "Test"])
        else:
            result = torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            #             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(["Train", "Valid", "Test"])

    def analysis_run(self, run, best_epoch, display_analysis=True):
        split_idx = self.split_idx_lst[run]
        train_idx, valid_idx, test_idx = (
            split_idx["train"].cpu().numpy(),
            split_idx["valid"].cpu().numpy(),
            split_idx["test"].cpu().numpy(),
        )

        train_idx = [
            True if idx in train_idx else False for idx in range(self.num_nodes)
        ]
        valid_idx = [
            True if idx in valid_idx else False for idx in range(self.num_nodes)
        ]
        test_idx = [True if idx in test_idx else False for idx in range(self.num_nodes)]

        num_train, num_valid, num_test = len(train_idx), len(valid_idx), len(test_idx)

        y_true = self.y_true
        y_pred = self.y_preds[run][best_epoch]
        is_labeled = y_true == y_true
        correct = y_true[is_labeled] == y_pred[is_labeled]

        unique_labels = np.unique(y_true)
        label_index = [y_true == label for label in unique_labels]
        correct_train_class = [
            y_pred[index & train_idx] == y_true[index & train_idx]
            for index in label_index
        ]
        correct_valid_class = [
            y_pred[index & valid_idx] == y_true[index & valid_idx]
            for index in label_index
        ]
        correct_test_class = [
            y_pred[index & test_idx] == y_true[index & test_idx]
            for index in label_index
        ]
        num_correct_train_class = np.array([np.sum(val) for val in correct_train_class])
        num_correct_valid_class = np.array([np.sum(val) for val in correct_valid_class])
        num_correct_test_class = np.array([np.sum(val) for val in correct_test_class])
        total_train_class = np.array(
            [np.sum(index & train_idx) for index in label_index]
        )
        total_valid_class = np.array(
            [np.sum(index & valid_idx) for index in label_index]
        )
        total_test_class = np.array([np.sum(index & test_idx) for index in label_index])

        if display_analysis:
            print("")
            for idx, label in enumerate(unique_labels):
                num_train_class, num_valid_class, num_test_class = (
                    total_train_class[idx],
                    total_valid_class[idx],
                    total_test_class[idx],
                )
                print(
                    f"Class {label + 1:02d}: train {num_correct_train_class[idx]:03d}/{num_train_class:03d} "
                    f"= {100 * num_correct_train_class[idx]/num_train_class:^6.2f} "
                    f"valid {num_correct_valid_class[idx]:03d}/{num_valid_class:03d} "
                    f"= {100 * num_correct_valid_class[idx]/num_valid_class:^6.2f} "
                    f"test {num_correct_test_class[idx]:03d}/{num_test_class:03d} "
                    f"= {100 * num_correct_test_class[idx]/num_test_class:^6.2f}"
                )
            print(
                f"Total     "
                f"train {num_correct_train_class.sum():03d}/{total_train_class.sum():03d} "
                f"= {100 * num_correct_train_class.sum() / total_train_class.sum() :^6.2f} "
                f"valid {num_correct_valid_class.sum():03d}/{total_valid_class.sum():03d} "
                f"= {100 * num_correct_valid_class.sum() / total_valid_class.sum() :^6.2f} "
                f"test {num_correct_test_class.sum():03d}/{total_test_class.sum():03d} "
                f"= {100 * num_correct_test_class.sum() / total_test_class.sum() :^6.2f}"
            )
        return None


def tune_hyperparameters(run, args, **kwargs):
    args.display_step = -1
    args.display_analysis = False
    args.debug = False

    for hps in product(*list(kwargs.values())):
        for key, value in zip(kwargs.keys(), hps):
            setattr(args, key, value)
        dataset, args = load_data(args)
        dataset, args = augmentation(args, dataset)
        dataset, args = preprocess_data(args, dataset)
        run(args, dataset)
