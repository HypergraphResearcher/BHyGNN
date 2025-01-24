import copy
import warnings

import numpy as np
from tqdm import tqdm
import time, os
import os.path as osp
from utils import *
from arguments import parse_args
from model import parse_model
import torch
import torch.nn.functional as F
import wandb
from model import BHyGNN

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_SILENT"] = "true"
warnings.filterwarnings("ignore", module="torch_geometric.data")


def run(args, dataset):

    if args.cuda in [0, 1, 2, 3]:
        device = torch.device(
            "cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cpu")
    # torch.set_default_device(device)
    model = BHyGNN(args)
    model = model.to(device)
    data = dataset.data.to(device)

    activation, loss_func, eval_func, dc_func = get_functions()

    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop
        )
        split_idx_lst.append(split_idx)

    runtime_list = []
    group_name = f"{args.model}_{args.dname}_{time.strftime('%Y-%m-%d %H:%M:%S')}"

    logger = Logger(args.runs, data.y.detach().cpu().numpy(), split_idx_lst)
    for run in tqdm(range(args.runs)):
        wandb.init(
            project="hgnn",
            group=group_name,
            name=f"{args.model}_{run}",
            config=args,
            job_type="train",
            mode="disabled" if args.debug else "online",
        )

        wandb.watch(model)
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx["train"].to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )

        for epoch in range(args.epochs):
            #         Training part
            model.train()
            optimizer.zero_grad()

            out = model(data=data)

            logit = activation(out, dim=1)

            if args.model == "model":
                loss_ce = loss_func(logit[train_idx], data.y[train_idx])
                loss_dc = dc_func(data.norm, model.stats, data.num_ori_edge, args.gamma)
                loss = (1 - args.alpha) * loss_ce + args.alpha * loss_dc
            else:
                loss_ce = loss_func(logit[train_idx], data.y[train_idx])
                loss_dc = 0
                loss = loss_ce

            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(data=data)
                logit = F.log_softmax(out, dim=1)

                result = evaluate(
                    y_true=data.y,
                    y_pred=logit,
                    split_idx=split_idx,
                    eval_func=eval_func,
                )

                logger.add_result(run, result)
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": result[3],
                        "valid_loss": result[4],
                        "test_loss": result[5],
                        "train_acc": result[0],
                        "valid_acc": result[1],
                        "test_acc": result[2],
                    }
                )

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(
                    f"Run: {run + 1:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Train Loss: {loss:.4f}, "
                    f"loss ce: {loss_ce:.4f}, "
                    f"loss dc: {loss_dc:.4f}, "
                    f"Valid Loss: {result[4]:.4f}, "
                    f"Test  Loss: {result[5]:.4f}, "
                    f"Train Acc: {result[0]:.2f}%, "
                    f"Valid Acc: {result[1]:.2f}%, "
                    f"Test  Acc: {result[2]:.2f}%"
                )

        end_time = time.time()
        runtime_list.append(end_time - start_time)
        best_epoch, result = logger.print_statistics(run, False)
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": loss,
                "valid_loss": result[4],
                "test_loss": result[5],
                "train_acc": result[0],
                "valid_acc": result[1],
                "test_acc": result[2],
            }
        )
        wandb.finish()
        logger.analysis_run(run, best_epoch, args.display_analysis)
    ### Save results ###

    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    best_epoch, best_result = logger.print_statistics()
    best_val, best_test = best_result[:, 1], best_result[:, 3]

    wandb.init(
        project="hgnn",
        group=group_name,
        name=f"{args.model}_{'_'.join([str(epoch) for epoch in best_epoch])}",
        job_type="summary",
        config=args,
        mode="disabled" if args.debug else "online",
    )

    wandb.log(
        {
            "Best Train": best_result[:, 0].mean(),
            "Final Valid": best_result[:, 1].mean(),
            "Final Train": best_result[:, 2].mean(),
            "Final Test": best_result[:, 3].mean(),
        }
    )
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    fix_seed(args.seed)
    dataset, args = load_data(args)
    dataset, args = augmentation(args, dataset)
    dataset, args = preprocess_data(args, dataset)
    run(args, dataset)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
