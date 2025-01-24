from typing import Optional

import torch, numpy as np
from torch import Tensor
import torch.nn as nn, torch.nn.functional as F
from .layers import glorot, MLP
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter
from torch_geometric.utils import softmax


class layer(MessagePassing):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers,
        dropout,
        Normalization="None",
        InputNorm=False,
        combine=True,
        attention=False,
        heads=1,
        negative_slope=0.2,
    ):
        super(layer, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.Normalization = Normalization
        self.InputNorm = InputNorm
        self.combine = combine
        self.attention = attention
        self.heads = heads
        self.node_dim = 0 if self.attention else -2
        self.attention_hidden = self.hid_dim // self.heads
        self.negative_slope = negative_slope

        if self.attention:
            if self.combine:
                self.f_enc = Linear(in_dim * 2, hid_dim)
                self.f_enc_k = Linear(in_dim * 2, hid_dim)
                self.f_dec = MLP(
                    hid_dim, hid_dim, out_dim, num_layers, 0.0, "None", False
                )
            else:
                self.f_enc = Linear(in_dim, hid_dim)
                self.f_enc_k = Linear(in_dim, hid_dim)
                self.f_dec = MLP(
                    hid_dim, hid_dim, out_dim, num_layers, 0.0, "None", False
                )
        else:
            if self.combine:
                if num_layers > 0:
                    self.f_enc = Linear(in_dim * 2, hid_dim)
                    # self.f_enc = MLP(in_dim*2, hid_dim, hid_dim, num_layers, 0.0, Normalization, InputNorm)
                    self.f_dec = MLP(
                        hid_dim,
                        hid_dim,
                        out_dim,
                        num_layers,
                        0.0,
                        Normalization,
                        InputNorm,
                    )
                else:
                    self.f_enc = Linear(in_dim * 2, hid_dim)
                    self.f_dec = nn.Identity()
            else:
                if num_layers > 0:
                    # self.f_enc = MLP(in_dim, hid_dim, hid_dim, num_layers, 0.0, Normalization, InputNorm)
                    self.f_enc = Linear(in_dim, hid_dim)
                    self.f_dec = MLP(
                        hid_dim,
                        hid_dim,
                        out_dim,
                        num_layers,
                        0.0,
                        Normalization,
                        InputNorm,
                    )
                else:
                    self.f_enc = nn.Identity()
                    self.f_dec = nn.Identity()

        self.ln0 = nn.LayerNorm(self.hid_dim)
        self.ln1 = nn.LayerNorm(self.hid_dim)
        self.att_r = Parameter(torch.Tensor(1, self.heads, self.attention_hidden))
        self.register_parameter("bias", None)

    def reset_parameters(self):
        if isinstance(self.f_enc, MLP):
            self.f_enc.reset_parameters()
        elif isinstance(self.f_enc, Linear):
            glorot(self.f_enc.weight)
        if isinstance(self.f_dec, MLP):
            self.f_dec.reset_parameters()
        elif isinstance(self.f_dec, Linear):
            glorot(self.f_dec.weight)
        # glorot(self.encoder.weight)
        # glorot(self.encoder_V.weight)
        # self.decoder.reset_parameters()
        if self.attention:
            if isinstance(self.f_enc_k, MLP):
                self.f_enc_k.reset_parameters()
            elif isinstance(self.f_enc_k, Linear):
                glorot(self.f_enc_k.weight)
            nn.init.xavier_uniform_(self.att_r)

        self.ln0.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, src, dst, edge_index, norm, aggr="add"):
        if self.attention:
            if self.combine:
                out = self.propagate(
                    edge_index, x=src, dst=dst, aggregate=aggr, norm=norm
                )
            else:
                out = self.propagate(
                    edge_index, x=src, dst=None, aggregate="add", norm=norm
                )
            out += self.att_r
        else:
            if self.combine:
                out = self.propagate(
                    edge_index, x=src, dst=dst, aggregate=aggr, norm=norm
                )
            else:
                out = self.propagate(
                    edge_index, x=src, dst=None, aggregate=aggr, norm=norm
                )
        out += self.att_r
        out = self.ln0(out.view(-1, self.hid_dim))
        out = self.ln1(out + F.relu(self.f_dec(out)))
        return out

    def message(self, x_j, dst_i, index, ptr, norm):
        if self.attention:
            num_nodes = index.max() + 1
            H, C = self.heads, self.attention_hidden
            x_j = torch.cat([x_j, dst_i], dim=-1) if self.combine else x_j
            x_K = self.f_enc_k(x_j).view(-1, H, C)
            x_V = self.f_enc(x_j).view(-1, H, C)
            alpha = (x_K * self.att_r).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, index, ptr, num_nodes)
            if norm is not None:
                alpha = norm.view(-1, 1) * alpha + 1e-6
                out_sum = scatter(alpha, index, dim=0, dim_size=num_nodes, reduce="sum")
                out_sum_index = out_sum.index_select(0, index)
                alpha = alpha / (out_sum_index + 1e-16)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            out = x_V * alpha.unsqueeze(-1)
        else:
            out = torch.cat([x_j, dst_i], dim=-1) if self.combine else x_j
            out = self.f_enc(out)
            out = F.relu(out)
            out = norm.view(-1, 1) * out
        return out

    def aggregate(self, inputs, index, dim_size=None, aggregate=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        if aggregate is None:
            raise ValueError("aggr was not passed!")
        # dim = self.node_dim if not self.attention else 0
        # dim = self.node_dim
        return scatter(inputs, index, dim=self.node_dim, reduce=aggregate)


class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
        self.args = args

        self.num_features = args.num_features
        self.num_classes = args.num_classes

        self.num_layers = args.num_layers
        self.num_MLP_layers = args.num_MLP_layers
        self.num_classifier_layers = args.num_classifier_layers
        self.num_MF_layers = args.num_MF_layers
        self.num_conv_layers = args.num_conv_layers

        self.input_dropout = args.input_dropout
        self.dropout = args.dropout
        self.InputNorm = args.input_norm
        self.normalization = args.normalization

        self.res = args.resnet
        self.aggr = args.aggregate
        self.tau = args.tau

        self.attention = args.attention
        self.heads = args.heads

        self.MLP_hidden = args.MLP_hidden
        self.classifier_hidden = args.classifier_hidden
        self.MF_hidden = args.MF_hidden

        self.V2ECB = args.V2ECB
        self.E2VCB = args.E2VCB

        self.GPR = args.GPR

        self.GPRweights = Linear(self.num_layers + 1, 1, bias=False)

        self.mapping = MLP(
            in_channels=self.num_features,
            hidden_channels=self.MLP_hidden,
            out_channels=self.MLP_hidden,
            num_layers=1,
            dropout=0,
            Normalization="None",
            InputNorm=False,
        )

        self.MLP = MLP(
            in_channels=self.MLP_hidden,
            hidden_channels=self.MLP_hidden,
            out_channels=self.MLP_hidden,
            num_layers=1,
            dropout=self.dropout,
            Normalization=self.normalization,
            InputNorm=False,
        )

        self.classifier = MLP(
            in_channels=self.MLP_hidden,
            hidden_channels=self.classifier_hidden,
            out_channels=self.num_classes,
            num_layers=args.num_classifier_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )

        self.V2EConv = nn.ModuleList()
        self.E2VConv = nn.ModuleList()

        for _ in range(self.num_layers):
            self.V2EConv.append(
                layer(
                    in_dim=self.MLP_hidden,
                    hid_dim=self.MLP_hidden,
                    out_dim=self.MLP_hidden,
                    num_layers=self.num_conv_layers,
                    dropout=self.dropout,
                    Normalization=self.normalization,
                    InputNorm=self.InputNorm,
                    combine=self.V2ECB,
                    attention=self.attention,
                    heads=self.heads,
                )
            )
            self.E2VConv.append(
                layer(
                    in_dim=self.MLP_hidden,
                    hid_dim=self.MLP_hidden,
                    out_dim=self.MLP_hidden,
                    num_layers=self.num_conv_layers,
                    dropout=self.dropout,
                    Normalization=self.normalization,
                    InputNorm=self.InputNorm,
                    combine=self.E2VCB,
                    attention=self.attention,
                    heads=self.heads,
                )
            )

        self.v2eVHGAE = VHGAE(
            src_in_dim=self.MLP_hidden,
            dst_in_dim=self.MLP_hidden,
            hid_dim=self.MF_hidden,
            out_dim=2,
            num_layers=self.num_MF_layers,
            dropout=self.dropout,
            Normalization=self.normalization,
            InputNorm=self.InputNorm,
            aggr=self.aggr,
            tau=self.tau,
        )

        self.e2vVHGAE = VHGAE(
            src_in_dim=self.MLP_hidden,
            dst_in_dim=self.MLP_hidden,
            hid_dim=self.MF_hidden,
            out_dim=2,
            num_layers=self.num_MF_layers,
            dropout=self.dropout,
            Normalization=self.normalization,
            InputNorm=self.InputNorm,
            aggr=self.aggr,
            tau=self.tau,
        )

    def reset_parameters(self):
        self.mapping.reset_parameters()
        self.MLP.reset_parameters()
        self.classifier.reset_parameters()
        self.v2eMF.reset_parameters()
        self.e2vMF.reset_parameters()
        self.v2eVHGAE.reset_parameters()
        self.e2vVHGAE.reset_parameters()
        self.GPRweights.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        norm = data.norm

        num_ori_edge = data.num_ori_edge

        cidx = edge_index[1].min()
        edge_index[1] -= cidx
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        self.norms = []
        self.stats = []

        # x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = F.relu(self.mapping(x))
        xs = [x]
        x_node = x
        x_he = scatter(x_node[edge_index[0]], edge_index[1], dim=-2, reduce=self.aggr)
        for i in range(self.num_layers):
            norm, stat = self.v2eVHGAE(x_node, x_he, edge_index, num_ori_edge)
            self.norms.append(norm)
            self.stats.append(stat)
            x_he = self.V2EConv[i](
                src=x_node, dst=x_he, edge_index=edge_index, norm=norm, aggr=self.aggr
            )
            x_he = F.relu(x_he)
            x_he = F.dropout(x_he, p=self.dropout, training=self.training)
            norm, stat = self.e2vVHGAE(x_he, x_node, reversed_edge_index, num_ori_edge)
            self.norms.append(norm)
            self.stats.append(stat)
            x_node = self.E2VConv[i](
                src=x_he,
                dst=x_node,
                edge_index=reversed_edge_index,
                norm=norm,
                aggr=self.aggr,
            )
            x_node = F.relu(x_node)
            xs.append(x_node)
            x_node = F.dropout(x_node, p=self.dropout, training=self.training)
            # x_node = (x_node * 0.5) + (x * 0.5) if self.res else x_node
            # x_node = self.MLP(x_node)
        if self.GPR:
            x_node = torch.stack(xs, dim=-1)
            x_node = self.GPRweights(x_node).squeeze()
        self.embeddings = x_node
        return self.classifier(x_node)


class VHGAE(nn.Module):
    def __init__(
        self,
        src_in_dim,
        dst_in_dim,
        hid_dim,
        out_dim,
        num_layers,
        dropout,
        Normalization="bn",
        InputNorm=False,
        aggr="add",
        tau=0.5,
    ):
        super(VHGAE, self).__init__()

        self.src_in_dim = src_in_dim
        self.dst_in_dim = dst_in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.Normalization = Normalization
        self.InputNorm = InputNorm
        self.aggr = aggr
        self.tau = tau

        assert self.out_dim == 2, "VHGAE only supports binary classification"

        self.src_encoder = MLP(
            in_channels=self.src_in_dim,
            hidden_channels=self.hid_dim,
            out_channels=self.hid_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            Normalization=self.Normalization,
            InputNorm=False,
        )

        self.dst_encoder = MLP(
            in_channels=self.dst_in_dim,
            hidden_channels=self.hid_dim,
            out_channels=self.hid_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            Normalization=self.Normalization,
            InputNorm=False,
        )

        self.decoder = MLP(
            in_channels=self.hid_dim,
            hidden_channels=self.hid_dim,
            out_channels=self.out_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            Normalization=self.Normalization,
            InputNorm=False,
        )

        self.mean_src = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim)
        )
        self.std_src = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Softplus(),
        )
        self.mean_dst = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim)
        )
        self.std_dst = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Softplus(),
        )

    def reset_parameters(self):
        self.decoder.reset_parameters()
        # self.encoder.reset_parameters()
        self.dst_encoder.reset_parameters()
        self.src_encoder.reset_parameters()
        for layer in self.mean_src:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.std_src:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.mean_dst:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.std_dst:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, src, dst, edge_index, num_ori_edge):
        num_self_loop = edge_index.shape[1] - num_ori_edge

        edge_index = edge_index[:, :num_ori_edge]

        device = src.device

        x_src, x_dst = src, dst

        x_src = self.src_encoder(x_src)
        x_dst = self.dst_encoder(x_dst)

        src_mean = self.mean_src(x_src)
        src_std = self.std_src(x_src)
        dst_mean = self.mean_dst(x_dst)
        dst_std = self.std_dst(x_dst)

        src_mask = torch.randn(src_mean.shape).to(device)
        x_src = src_mask * src_std + src_mean

        dst_mask = torch.randn(dst_mean.shape).to(device)
        x_dst = dst_mask * dst_std + dst_mean

        prob = x_src[edge_index[0]] * x_dst[edge_index[1]]

        prob = self.decoder(prob).squeeze()
        logit = F.gumbel_softmax(prob, tau=self.tau, hard=True)
        pred = logit[:, 1]

        self_loop = torch.ones(num_self_loop, dtype=torch.float32).to(edge_index.device)
        out = torch.cat([pred, self_loop])

        stat = {
            "x_src": x_src,
            "x_dst": x_dst,
            "src_mean": src_mean,
            "src_std": src_std,
            "dst_mean": dst_mean,
            "dst_std": dst_std,
            "pred": out,
            "GAE": True,
            "logit": logit,
        }

        return out, stat
