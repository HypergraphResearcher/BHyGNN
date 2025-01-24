import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))
    return 100 * sum(acc_list) / len(acc_list)
    # f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    # return f1 * 100


def get_activation_function():
    return F.log_softmax


def get_loss_function():
    return F.nll_loss


def get_evaluation_function():
    return eval_acc


def get_functions():
    activation = get_activation_function()
    loss_func = get_loss_function()
    eval_func = get_evaluation_function()
    dc_func = get_dc_function()

    return activation, loss_func, eval_func, dc_func


def dc_func(y_true, stats, num_ori_edge, ratio):
    # todo regulation    L1  / L2

    loss_rec = 0
    kl_div_src, kl_div_dst = 0, 0

    len_stats = len(stats)

    y_true = y_true[:num_ori_edge]

    for stat in stats:

        x_src, x_dst = stat["x_src"], stat["x_dst"]
        x_mean_src, x_std_src, x_mean_dst, x_std_dst = (
            stat["src_mean"],
            stat["src_std"],
            stat["dst_mean"],
            stat["dst_std"],
        )

        pred = stat["pred"][:num_ori_edge]
        # logit = stat["logit"][:num_ori_edge]
        # loss_rec += F.binary_cross_entropy(input=pred[:num_ori_edge],
        #                                    target=edge_index_pos[:num_ori_edge].to(torch.float32))
        # loss_rec += F.binary_cross_entropy(input=pred_neg,
        #                                    target=torch.zeros(edge_index_neg.shape[1]).to(edge_index_neg.device))
        # loss_rec += dc.sum()
        # loss_rec +=  torch.square(y_true[:num_ori_edge] - pred[:num_ori_edge]).mean() * ratio

        loss_rec += torch.abs(y_true.sum() * ratio - pred.sum()) / y_true.sum()

        # loss_rec += pred[:num_ori_edge].sum()
        # loss_rec += torch.abs(y_true-pred).mean()

        # loss_rec += F.cross_entropy(pred, y_true) / y_true.shape[0]
        # loss_rec += F.mse_loss(pred, y_true)

        if stat["GAE"]:
            kl_div_src += (
                -0.5
                * (1 + 2 * torch.log(x_std_src) - x_mean_src**2 - x_std_src**2)
                .sum(dim=1)
                .mean()
            )
            kl_div_dst += (
                -0.5
                * (1 + 2 * torch.log(x_std_dst) - x_mean_dst**2 - x_std_dst**2)
                .sum(dim=1)
                .mean()
            )

    # return (loss_rec / len_stats + kl_div_src + kl_div_dst).mean()

    return loss_rec + kl_div_src + kl_div_dst
    # return loss_rec

    # loss_rec = 0

    # for dc in dcs:
    #     # loss_rec +=  torch.abs(ratio * y_true.sum() - dc.sum())
    #     loss_rec += dc.sum() * ratio
    # return loss_rec/len(dcs)


def get_dc_function():
    return dc_func
