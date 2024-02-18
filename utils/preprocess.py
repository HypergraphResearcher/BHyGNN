from itertools import combinations
import numpy as np
import torch
import torch_sparse
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from collections import defaultdict, Counter
import os.path as osp
import networkx as nx
from sklearn.cluster import AgglomerativeClustering

from sklearn import metrics


def GenerateHomo(data):
    V2E = data.edge_index.cpu().numpy()
    y = data.y.cpu().numpy()
    num_classes = data.num_classes
    v2e, e2v = defaultdict(list), defaultdict(list)
    for i in range(len(V2E[0])):
        v2e[V2E[0][i]].append(V2E[1][i])
        e2v[V2E[1][i]].append(V2E[0][i])

    he_homo = []
    he_class = []
    for he, nodes in e2v.items():
        count_dict = {label: 0 for label in range(num_classes)}
        for node in nodes:
            y_node = y[node].item()
            count_dict[y_node] += 1
        majority_class = max(count_dict, key=count_dict.get)
        he_homo.append(count_dict[majority_class] / len(nodes))
        he_class.append(majority_class)
    node_homo = []
    for node, hes in v2e.items():
        nodes_propagate = []
        y_node = y[node].item()
        for he in hes:
            nodes_propagate.extend(e2v[he])
        num_same = 0
        for node_p in nodes_propagate:
            y_node_p = y[node_p].item()
            if y_node_p == y_node:
                num_same += 1
        node_homo.append(num_same / len(nodes_propagate))
    data.homo_he = torch.FloatTensor(he_homo)
    data.homo_node = torch.FloatTensor(node_homo)
    data.homo_class = torch.LongTensor(he_class)
    return data


def IRMM(args, data, threshold=0.01, recompute=False):
    X = data.x.numpy()
    num_classes = args.num_classes
    num_nodes, num_hyperedges = data.num_nodes, data.num_hyperedges
    H = data.edge_index.to_dense().numpy()
    file_path = osp.join(args.root_dir, "data", "exp", f"IRMM_{args.dname}.txt")

    if osp.exists(file_path) and not recompute:
        A = np.loadtxt(file_path)
        A = A + np.eye(num_nodes)
        A = NPNormalize(A)
        A = ndarry_to_torch_sparse_tensor(A)
        data.edge_index = A
    else:
        Dv, De = np.diag(H.sum(1)), np.diag(H.sum(0))
        Iv, Ie = np.eye(num_nodes), np.eye(num_hyperedges)

        agg_cluster = AgglomerativeClustering(n_clusters=num_classes)

        W = np.eye(num_hyperedges)
        delta = threshold + 1
        while delta >= threshold:
            inv_De_Ie = np.linalg.inv(De - Ie)
            HW = np.dot(H, W)

            A = np.dot(np.dot(HW, inv_De_Ie), H.T)
            np.fill_diagonal(A, 0)

            G = nx.from_numpy_array(A)
            clusters = nx.community.louvain_communities(G, seed=args.seed)

            centroids = np.array(
                [X[list(cluster)].mean(axis=0) for cluster in clusters]
            )

            agg_cluster = AgglomerativeClustering(n_clusters=num_classes)
            agg_cluster.fit(centroids)

            cluster_labels = agg_cluster.labels_

            agg_clusters = [[] for _ in range(num_classes)]
            for cluster, label in zip(clusters, cluster_labels):
                agg_clusters[label].extend(list(cluster))

            dW = np.zeros(num_hyperedges)
            for e_idx, e in enumerate(H.T):
                nodes = e.nonzero()[0]
                k = [0 for _ in range(num_classes)]
                for class_label, class_cluster in enumerate(agg_clusters):
                    k[class_label] = len(np.intersect1d(nodes, class_cluster))

                dW[e_idx] = (1 / num_hyperedges) * sum(
                    [
                        (1 / (k[i] + 1)) * (len(nodes) + num_classes)
                        for i in range(num_classes)
                    ]
                )

            W_pre = W
            W = 0.5 * (W_pre + dW.reshape(-1, 1))

            delta = np.linalg.norm(W - W_pre)
            print(delta)

        inv_De_Ie = np.linalg.inv(De - Ie)
        HW = np.dot(H, W)
        A = np.dot(np.dot(HW, inv_De_Ie), H.T)

        np.savetxt(file_path, A)

        A = A + np.eye(num_nodes)
        A = torch.FloatTensor(A)
        data.edge_index = A

    # A = np.loadtxt(file_path)
    #
    # G = nx.from_numpy_array(A)
    # clusters = nx.community.louvain_communities(G, weight="weight")
    #
    # centroids = np.array([X[list(cluster)].mean(axis=0) for cluster in clusters])
    #
    # agg_cluster = AgglomerativeClustering(n_clusters=num_classes)
    # agg_cluster.fit(centroids)
    #
    # cluster_labels = agg_cluster.labels_
    #
    # agg_clusters = [[] for _ in range(num_classes)]
    # for cluster, label in zip(clusters, cluster_labels):
    #     agg_clusters[label].extend(list(cluster))
    #
    # y_true = data.y.numpy()
    # y_pred = np.zeros(num_nodes)
    # for idx, cluster in enumerate(agg_clusters):
    #     y_pred[cluster] = idx

    return data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def ndarry_to_torch_sparse_tensor(mx):
    row, col = mx.nonzero()
    indices = torch.from_numpy(np.vstack((row, col))).long()
    values = mx[row, col]
    shape = torch.Size(mx.shape)

    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def NPNormalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = (r_mat_inv).dot(mx)
    return mx


def ConstructLE(data, v_threshold=30, e_threshold=30, sparse=False):
    """construct line expansion from original hypergraph
    INPUT:
        - pairs <matrix>
            - size: N x 2. N means the total vertex-hyperedge pair of the hypergraph
            - each row contains the idx_of_vertex, idx_of_hyperedge
        - v_threshold: vertex-similar neighbor sample threshold
        - e_threshold: hyperedge-similar neighbor sample threshold
    Concept:
        - vertex, hyperedge: for the hypergraph
        - node, edge: for the induced simple graph
    OUTPUT:
        - adj <sparse coo_matrix>: N_node x N_node
        - Pv <sparse coo_matrix>: N_node x N_vertex
        - PvT <sparse coo_matrix>: N_vertex x N_node
        - Pe <sparse coo_matrix>: N_node x N_hyperedge
        - PeT <sparse coo_matrix>: N_hyperedge x N_node
    """

    # get # of vertices and encode them starting from 0

    edge_index = data.edge_index.numpy().T

    uniq_vertex = np.unique(edge_index[:, 0])

    isolated_vertex = np.setdiff1d(np.arange(data.x.size(0)), uniq_vertex)
    isolated_hyperedge_index = edge_index[:, 1].max() + 1
    isolated_edges = np.stack(
        (
            isolated_vertex,
            np.arange(
                isolated_hyperedge_index,
                isolated_hyperedge_index + isolated_vertex.shape,
            ),
        ),
        axis=1,
    )

    edge_index = np.vstack((edge_index, isolated_edges))

    uniq_vertex = np.unique(edge_index[:, 0])
    N_vertex = len(uniq_vertex)

    # assert N_vertex == data.x.size(0), "Some node does not connect to any hyperedge"

    edge_index[:, 0] = list(
        map({vertex: i for i, vertex in enumerate(uniq_vertex)}.get, edge_index[:, 0])
    )

    # get # of hyperedges and encode them starting from 0
    uniq_hyperedge = np.unique(edge_index[:, 1])
    N_hyperedge = len(uniq_hyperedge)
    edge_index[:, 1] = list(
        map(
            {hyperedge: i for i, hyperedge in enumerate(uniq_hyperedge)}.get,
            edge_index[:, 1],
        )
    )

    N_node = edge_index.shape[0]

    # vertex projection: from vertex to node
    Pv = sp.coo_matrix(
        (np.ones(N_node), (np.arange(N_node), edge_index[:, 0])),
        shape=(N_node, N_vertex),
        dtype=np.float32,
    )  # (N_node, N_vertex)
    # vertex back projection (Pv Transpose): from node to vertex

    weight = np.ones(N_node)
    for vertex in range(N_vertex):
        tmp = np.where(edge_index[:, 0] == vertex)[0]
        weight[tmp] = 1.0 / len(tmp)
    PvT = sp.coo_matrix(
        (weight, (edge_index[:, 0], np.arange(N_node))),
        shape=(N_vertex, N_node),
        dtype=np.float32,
    )  # (N_vertex, N_node)

    # hyperedge projection: from hyperedge to node
    Pe = sp.coo_matrix(
        (np.ones(N_node), (np.arange(N_node), edge_index[:, 1])),
        shape=(N_node, N_hyperedge),
        dtype=np.float32,
    )  # (N_node, N_hyperedge)
    # hyperedge back projection (Pe Transpose): from node to hyperedge
    weight = np.ones(N_node)
    for hyperedge in range(N_hyperedge):
        tmp = np.where(edge_index[:, 1] == hyperedge)[0]
        weight[tmp] = 1.0 / len(tmp)
    PeT = sp.coo_matrix(
        (weight, (edge_index[:, 1], np.arange(N_node))),
        shape=(N_hyperedge, N_node),
        dtype=np.float32,
    )  # (N_node, N_hyperedge)

    # construct adj
    edges = []
    # get vertex-similar edges
    for vertex in range(N_vertex):
        position = np.where(edge_index[:, 0] == vertex)[0]
        if len(position) > v_threshold:
            position = np.random.choice(position, v_threshold, replace=False)
            tmp_edge = np.array(list(combinations(position, r=2)))
            edges += list(tmp_edge)
        else:
            edges += list(combinations(position, r=2))

    # get hyperedge-similar edges
    for hyperedge in range(N_hyperedge):
        position = np.where(edge_index[:, 1] == hyperedge)[0]
        if len(position) > e_threshold:
            position = np.random.choice(position, e_threshold, replace=False)
            tmp_edge = np.array(list(combinations(position, r=2)))
            edges += list(list(tmp_edge))
        else:
            edges += list(combinations(position, r=2))

    edges = np.array(edges)

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(N_node, N_node),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = NPNormalize(adj + 2.0 * sp.eye(adj.shape[0]))

    data.x = torch.FloatTensor(Pv.todense()) @ data.x

    if sparse:
        data.PvT = sparse_mx_to_torch_sparse_tensor(PvT)
        data.edge_index = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        data.PvT = torch.FloatTensor(PvT.todense())
        data.edge_index = torch.FloatTensor(adj.todense())
    # return adj, Pv, PvT, Pe, PeT
    return data


def get_HyperGCN_He_dict(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.edge_index)
    """
    For each he, clique-expansion. Note that we allow the weighted edge.
    Note that if node pair (vi,vj) is contained in both he1, he2, we will have (vi,vj) twice in edge_index. (weighted version CE)
    We default no self loops so far.
    """
    # #     Construct a dictionary
    #     He2V_List = []
    # #     Sort edge_index according to he_id
    #     _, sorted_idx = torch.sort(edge_index[1])
    #     edge_index = edge_index[:,sorted_idx].type(torch.LongTensor)
    #     current_heid = -1
    #     for idx, he_id in enumerate(edge_index[1]):
    #         if current_heid != he_id:
    #             current_heid = he_id
    #             if idx != 0 and len(he2v)>1: #drop original self loops
    #                 He2V_List.append(he2v)
    #             he2v = []
    #         he2v.append(edge_index[0,idx].item())
    # #     Remember to append the last he
    #     if len(he2v)>1:
    #         He2V_List.append(he2v)
    # #     Now, turn He2V_List into a dictionary
    edge_index[1, :] = edge_index[1, :] - edge_index[1, :].min()
    He_dict = {}
    for he in np.unique(edge_index[1, :]):
        #         ipdb.set_trace()
        nodes_in_he = list(edge_index[0, :][edge_index[1, :] == he])
        He_dict[he.item()] = nodes_in_he

    #     for he_id, he in enumerate(He2V_List):
    #         He_dict[he_id] = he

    return He_dict


def rand_train_test_idx(
    label, train_prop=0.5, valid_prop=0.25, ignore_negative=True, balance=False
):
    """Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num : train_num + valid_num]
        test_indices = perm[train_num + valid_num :]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    else:
        #         ipdb.set_trace()
        indices = []
        for i in range(label.max() + 1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop / (label.max() + 1) * len(label))
        val_lb = int(valid_prop * len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    return split_idx


def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
    #     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.num_nodes
    num_hyperedges = data.num_hyperedges
    if not ((num_nodes + num_hyperedges - 1) == data.edge_index[0].max().item()):
        raise ValueError("num_hyperedges does not match! 1")
    cidx = torch.where(edge_index[0] == num_nodes)[0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    if hasattr(data, "edge_y"):
        data.edge_y = data.edge_y[sorted_idx][:cidx]
    return data


def ConstructV2V(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.edge_index)
    """
    For each he, clique-expansion. Note that we DONT allow duplicated edges.
    Instead, we record its corresponding weights.
    We default no self loops so far.
    """
    # # Note that the method below for CE can be memory expensive!!!
    #     new_edge_index = []
    #     for he in np.unique(edge_index[1, :]):
    #         nodes_in_he = edge_index[0, :][edge_index[1, :] == he]
    #         if len(nodes_in_he) == 1:
    #             continue #skip self loops
    #         combs = combinations(nodes_in_he,2)
    #         for comb in combs:
    #             new_edge_index.append([comb[0],comb[1]])

    #     new_edge_index, new_edge_weight = torch.tensor(new_edge_index).type(torch.LongTensor).unique(dim=0,return_counts=True)
    #     data.edge_index = new_edge_index.transpose(0,1)
    #     data.norm = new_edge_weight.type(torch.float)

    # # Use the method below for better memory complexity
    edge_weight_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = np.sort(edge_index[0, :][edge_index[1, :] == he])
        if len(nodes_in_he) == 1:
            continue  # skip self loops
        combs = combinations(nodes_in_he, 2)
        for comb in combs:
            if not comb in edge_weight_dict.keys():
                edge_weight_dict[comb] = 1
            else:
                edge_weight_dict[comb] += 1

    # # Now, translate dict to edge_index and norm
    #
    new_edge_index = np.zeros((2, len(edge_weight_dict)))
    new_norm = np.zeros((len(edge_weight_dict)))
    cur_idx = 0
    for edge in edge_weight_dict:
        new_edge_index[:, cur_idx] = edge
        new_norm[cur_idx] = edge_weight_dict[edge]
        cur_idx += 1

    data.edge_index = torch.tensor(new_edge_index).type(torch.LongTensor)
    data.norm = torch.tensor(new_norm).type(torch.FloatTensor)
    return data


def ConstructH(data, override=False):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
    #     ipdb.set_trace()
    edge_index = np.array(data.edge_index)
    # Don't use edge_index[0].max()+1, as some nodes maybe isolated
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(edge_index[1]) - np.min(edge_index[1]) + 1
    assert num_hyperedges == data.num_hyperedges
    H = np.zeros((num_nodes, num_hyperedges))
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.0
        cur_idx += 1
    if override:
        data.edge_index = H
    data.H = H
    return data


def GenerateV2E(data):
    """
    Generate hyperedge features through nodes
    """
    H, X = data.H.numpy(), data.x.numpy()
    edge_features = np.dot(H.T, X)
    data.edge_features = torch.FloatTensor(edge_features)
    return data


def Add_Self_Loops(data):
    # update so we dont jump on some indices
    # Assume edge_index = [V;E]. If not, use ExtractV2E()
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_hyperedges = data.num_hyperedges
    data.num_ori_edge = edge_index.shape[1]

    if not (
        (data.num_nodes + data.num_hyperedges - 1) == data.edge_index[1].max().item()
    ):
        raise ValueError("num_hyperedges does not match! 2")

    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    # store the nodes that already have self-loops
    # skip_node_lst = []
    # for edge in hyperedge_appear_fre:
    #     if hyperedge_appear_fre[edge] == 1:
    #         skip_node = edge_index[0][torch.where(
    #             edge_index[1] == edge)[0].item()]
    #         skip_node_lst.append(skip_node.item())
    skip_node_lst = set()
    for edge in hyperedge_appear_fre:
        if hyperedge_appear_fre[edge] == 1:
            skip_node = edge_index[0][torch.where(edge_index[1] == edge)[0].item()]
            skip_node_lst.add(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    new_edges = torch.zeros((2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)
    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.num_hyperedges = num_hyperedges + num_nodes - len(skip_node_lst)
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    return data


def expand_edge_index(data, edge_th=0):
    """
    args:
        num_nodes: regular nodes. i.e. x.shape[0]
        num_edges: number of hyperedges. not the star expansion edges.

    this function will expand each n2he relations, [[n_1, n_2, n_3],
                                                    [e_7, e_7, e_7]]
    to :
        [[n_1,   n_1,   n_2,   n_2,   n_3,   n_3],
         [e_7_2, e_7_3, e_7_1, e_7_3, e_7_1, e_7_2]]

    and each he2n relations:   [[e_7, e_7, e_7],
                                [n_1, n_2, n_3]]
    to :
        [[e_7_1, e_7_2, e_7_3],
         [n_1,   n_2,   n_3]]

    and repeated for every hyperedge.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    if hasattr(data, "totedges"):
        num_edges = data.totedges
    else:
        num_edges = data.num_hyperedges[0]

    expanded_n2he_index = []
    #     n2he_with_same_heid = []

    #     expanded_he2n_index = []
    #     he2n_with_same_heid = []

    # start edge_id from the largest node_id + 1.
    cur_he_id = num_nodes
    # keep an mapping of new_edge_id to original edge_id for edge_size query.
    new_edge_id_2_original_edge_id = {}

    # do the expansion for all annotated he_id in the original edge_index
    #     ipdb.set_trace()
    for he_idx in range(num_nodes, num_edges + num_nodes):
        # find all nodes within the same hyperedge.
        selected_he = edge_index[:, edge_index[1] == he_idx]
        size_of_he = selected_he.shape[1]

        #         Trim a hyperedge if its size>edge_th
        if edge_th > 0:
            if size_of_he > edge_th:
                continue

        if size_of_he == 1:
            # there is only one node in this hyperedge -> self-loop node. add to graph.
            #             n2he_with_same_heid.append(selected_he)

            new_n2he = selected_he.clone()
            new_n2he[1] = cur_he_id
            expanded_n2he_index.append(new_n2he)

            # ====
            #             new_he2n_same_heid = torch.flip(selected_he, dims = [0])
            #             he2n_with_same_heid.append(new_he2n_same_heid)

            #             new_he2n = torch.flip(selected_he, dims = [0])
            #             new_he2n[0] = cur_he_id
            #             expanded_he2n_index.append(new_he2n)

            cur_he_id += 1
            continue

        # -------------------------------
        #         # new_n2he_same_heid uses same he id for all nodes.
        #         new_n2he_same_heid = selected_he.repeat_interleave(size_of_he - 1, dim = 1)
        #         n2he_with_same_heid.append(new_n2he_same_heid)

        # for new_n2he mapping. connect the nodes to all repeated he first.
        # then remove those connection that corresponding to the node itself.
        new_n2he = selected_he.repeat_interleave(size_of_he, dim=1)

        # new_edge_ids start from the he_id from previous iteration (cur_he_id).
        new_edge_ids = torch.LongTensor(
            np.arange(cur_he_id, cur_he_id + size_of_he)
        ).repeat(size_of_he)
        new_n2he[1] = new_edge_ids

        # build a mapping between node and it's corresponding edge.
        # e.g. {n_1: e_7_1, n_2: e_7_2}
        tmp_node_id_2_he_id_dict = {}
        for idx in range(size_of_he):
            new_edge_id_2_original_edge_id[cur_he_id] = he_idx
            cur_node_id = selected_he[0][idx].item()
            tmp_node_id_2_he_id_dict[cur_node_id] = cur_he_id
            cur_he_id += 1

        # create n2he by deleting the self-product edge.
        new_he_select_mask = torch.BoolTensor([True] * new_n2he.shape[1])
        for col_idx in range(new_n2he.shape[1]):
            tmp_node_id, tmp_edge_id = (
                new_n2he[0, col_idx].item(),
                new_n2he[1, col_idx].item(),
            )
            if tmp_node_id_2_he_id_dict[tmp_node_id] == tmp_edge_id:
                new_he_select_mask[col_idx] = False
        new_n2he = new_n2he[:, new_he_select_mask]
        expanded_n2he_index.append(new_n2he)

    #         # ---------------------------
    #         # create he2n from mapping.
    #         new_he2n = np.array([[he_id, node_id] for node_id, he_id in tmp_node_id_2_he_id_dict.items()])
    #         new_he2n = torch.from_numpy(new_he2n.T).to(device = edge_index.device)
    #         expanded_he2n_index.append(new_he2n)

    #         # create he2n with same heid as input edge_index.
    #         new_he2n_same_heid = torch.zeros_like(new_he2n, device = edge_index.device)
    #         new_he2n_same_heid[1] = new_he2n[1]
    #         new_he2n_same_heid[0] = torch.ones_like(new_he2n[0]) * he_idx
    #         he2n_with_same_heid.append(new_he2n_same_heid)

    new_edge_index = torch.cat(expanded_n2he_index, dim=1)
    #     new_he2n_index = torch.cat(expanded_he2n_index, dim = 1)
    #     new_edge_index = torch.cat([new_n2he_index, new_he2n_index], dim = 1)
    # sort the new_edge_index by first row. (node_ids)
    new_order = new_edge_index[0].argsort()
    data.edge_index = new_edge_index[:, new_order]

    return data


def norm_contruction(data, option="all_one", TYPE="V2E"):
    if TYPE == "V2E":
        if option == "all_one":
            data.norm = torch.ones_like(data.edge_index[0])

        elif option == "deg_half_sym":
            edge_weight = torch.ones_like(data.edge_index[0])
            cidx = data.edge_index[1].min()
            Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
            HEdeg = scatter_add(edge_weight, data.edge_index[1] - cidx, dim=0)
            V_norm = Vdeg ** (-1 / 2)
            E_norm = HEdeg ** (-1 / 2)
            data.norm = V_norm[data.edge_index[0]] * E_norm[data.edge_index[1] - cidx]

    elif TYPE == "V2V":
        data.edge_index, data.norm = gcn_norm(
            data.edge_index, data.norm, add_self_loops=True
        )
    return data


def GenerateDist(data, TYPE="euclidean"):
    X = data.x.numpy()

    num_nodes = data.num_nodes

    if TYPE == "euclidean":
        distances = cdist(X, X, TYPE)
    elif TYPE == "cosine":
        distances = cdist(X, X, TYPE)
        distances = (distances + 1) / 2
        distances += 1e-5
    # distances = np.ones((X.shape[0], X.shape[0]))
    # sorted_indices = np.argsort(distances, axis=1)
    # kth_smallest_values = distances[np.arange(X.shape[0]), sorted_indices[:, K - 1]]
    # kth_smallest_values += 1e-5
    data.dists = distances

    return data


def GenerateSTAR(data):
    X = data.x.numpy()
    H = data.H.T
    edge_index = data.edge_index.numpy().T

    num_hyperedges = data.num_hyperedges
    num_nodes = data.num_nodes

    hyperedge_X = H @ X / H.sum(1).reshape(-1, 1)

    X = np.concatenate((X, hyperedge_X))

    PvT = sp.coo_matrix(
        (np.ones(num_nodes), (np.arange(num_nodes), np.arange(num_nodes))),
        shape=(num_nodes, num_nodes + num_hyperedges),
        dtype=np.float32,
    )
    PvT = sparse_mx_to_torch_sparse_tensor(PvT)

    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[0]), (edge_index[:, 0], edge_index[:, 1])),
        shape=(num_nodes + num_hyperedges, num_nodes + num_hyperedges),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = NPNormalize(adj + 2.0 * sp.eye(adj.shape[0]))

    data.edge_index = sparse_mx_to_torch_sparse_tensor(adj)
    data.PvT = PvT
    data.x = torch.FloatTensor(X)

    return data


def generate_G_from_H(data):
    """
    This function generate the propagation matrix G for HGNN from incidence matrix H.
    Here we assume data.edge_index is already the incidence matrix H. (can be done by ConstructH())
    Adapted from HGNN github repo: https://github.com/iMoonLab/HGNN
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    #     ipdb.set_trace()
    H = data.edge_index
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    #     replace nan with 0. This is caused by isolated nodes
    DV2 = np.nan_to_num(DV2)
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    #     if variable_weight:
    #         DV2_H = DV2 * H
    #         invDE_HT_DV2 = invDE * HT * DV2
    #         return DV2_H, W, invDE_HT_DV2
    #     else:
    G = DV2 * H * W * invDE * HT * DV2
    data.edge_index = torch.Tensor(G)
    return data


def ConstructH_HNHN(data):
    """
    Construct incidence matrix H of size (num_nodes, num_hyperedges) from edge_index = [V;E]
    """
    edge_index = np.array(data.edge_index)
    num_nodes = data.num_nodes
    num_hyperedges = int(data.num_hyperedges)
    H = np.zeros((num_nodes, num_hyperedges))
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.0
        cur_idx += 1

    #     data.incident_mat = H
    return H


def generate_G_for_HNHN(data, args):
    """
    This function generate the propagation matrix G_V2E and G_E2V for HNHN from incidence matrix H.
    Here we assume data.edge_index is already the incidence matrix H. (can be done by ConstructH())

    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    #     ipdb.set_trace()
    H = data.edge_index
    alpha = args.HNHN_alpha
    beta = args.HNHN_beta
    H = np.array(H)

    # the degree of the node
    DV = np.sum(H, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    G_V2E = np.diag(DE ** (-beta)) @ H.T @ np.diag(DV ** (beta))
    G_E2V = np.diag(DV ** (-alpha)) @ H @ np.diag(DE ** (alpha))

    #     if variable_weight:
    #         DV2_H = DV2 * H
    #         invDE_HT_DV2 = invDE * HT * DV2
    #         return DV2_H, W, invDE_HT_DV2
    #     else:
    data.G_V2E = torch.Tensor(G_V2E)
    data.G_E2V = torch.Tensor(G_E2V)
    return data


def generate_norm_HNHN(H, data, args):
    """
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    #     H = data.incident_mat
    alpha = args.HNHN_alpha
    beta = args.HNHN_beta
    H = np.array(H)

    # the degree of the node
    DV = np.sum(H, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    num_nodes = data.num_nodes
    num_hyperedges = int(data.num_hyperedges)
    # alpha part
    D_e_alpha = DE**alpha
    D_v_alpha = np.zeros(num_nodes)
    for i in range(num_nodes):
        # which edges this node is in
        he_list = np.where(H[i] == 1)[0]
        D_v_alpha[i] = np.sum(DE[he_list] ** alpha)

    # beta part
    D_v_beta = DV**beta
    D_e_beta = np.zeros(num_hyperedges)
    for i in range(num_hyperedges):
        # which nodes are in this hyperedge
        node_list = np.where(H[:, i] == 1)[0]
        D_e_beta[i] = np.sum(DV[node_list] ** beta)

    D_v_alpha_inv = 1.0 / D_v_alpha
    D_v_alpha_inv[D_v_alpha_inv == float("inf")] = 0

    D_e_beta_inv = 1.0 / D_e_beta
    D_e_beta_inv[D_e_beta_inv == float("inf")] = 0

    data.D_e_alpha = torch.from_numpy(D_e_alpha).float()
    data.D_v_alpha_inv = torch.from_numpy(D_v_alpha_inv).float()
    data.D_v_beta = torch.from_numpy(D_v_beta).float()
    data.D_e_beta_inv = torch.from_numpy(D_e_beta_inv).float()

    return data


def GenerateHetero(data, add_self_loop):
    edge_index = data.edge_index

    edge_type = data.edge_type
    num_nodes, num_hyperedges = data.num_nodes, data.num_hyperedges
    num_relation_types = data.num_relation_types

    assert (edge_type.max() - edge_type.min()) == num_relation_types

    edge_relation = []

    for relation_label in range(num_relation_types):
        hyperedge_relation_idx = (edge_type == relation_label).nonzero(as_tuple=True)[0]
        hyperedge_relation_idx += num_nodes

        relation_edge_idx = torch.stack(
            [(edge_index[1] == index) for index in hyperedge_relation_idx]
        ).sum(0)

        if add_self_loop:
            relation_edge_index = edge_index[
                :, relation_edge_idx.nonzero(as_tuple=True)[0]
            ]
            hyperedge_appear_fre = Counter(relation_edge_index[1].numpy())
            skip_node_lst = set()
            for edge in hyperedge_appear_fre:
                if hyperedge_appear_fre[edge] == 1:
                    skip_node = relation_edge_index[0][
                        torch.where(relation_edge_index[1] == edge)[0].item()
                    ]
                    skip_node_lst.add(skip_node.item())

            add_edge_idx = torch.LongTensor(
                [
                    1 if node_idx not in skip_node_lst else 0
                    for node_idx in range(num_nodes)
                ]
            )
            relation_edge_idx = torch.cat((relation_edge_idx, add_edge_idx), dim=0)
        edge_relation.append(relation_edge_idx)
    edge_relation = torch.stack(edge_relation)
    if add_self_loop:
        new_edge_idx = edge_index[1].max() + 1
        new_edges = torch.stack(
            (
                torch.arange(0, num_nodes),
                torch.arange(new_edge_idx, new_edge_idx + num_nodes),
            )
        )
        edge_index = torch.cat((edge_index, new_edges), dim=1)
        data.edge_index = edge_index
    data.edge_relation = edge_relation > 0
    return data


def GenerateSnapshot(data):
    s = 1.0
    num_relation_types = data.num_relation_types
    label_edge = data.label_edge
    data.snapshots = []
    for relation_type_idx in range(num_relation_types):
        hyperedge_relation_idx = (label_edge[:, relation_type_idx] == 1).nonzero(
            as_tuple=True
        )[0]
        H = data.H[:, hyperedge_relation_idx].to(torch.float32)
        W_e_diag = torch.ones(H.shape[1])

        D_e_diag = torch.sum(H, 0)
        D_e_diag = D_e_diag.view((D_e_diag.shape[0])) + 1e-5

        D_v_diag = torch.mm(H, W_e_diag.view((W_e_diag.shape[0]), 1))
        D_v_diag = D_v_diag.view((D_v_diag.shape[0])) + 1e-5

        D_v_diag_inverse_sqrt = torch.diag(torch.pow(D_v_diag, -0.5))
        D_e_diag_inverse = torch.diag(torch.pow(D_e_diag, -1))

        Theta = (
            D_v_diag_inverse_sqrt
            @ H
            @ torch.diag(W_e_diag)
            @ D_e_diag_inverse
            @ H.T
            @ D_v_diag_inverse_sqrt
        )

        # Theta = torch.diag(torch.pow(D_v_diag, -0.5)) @ \
        #         H @ torch.diag(W_e_diag) @ \
        #         torch.diag(torch.pow(D_e_diag, -1)) @ \
        #         H.T @ \
        #         torch.diag(torch.pow(D_v_diag, -0.5))

        Theta_inverse = torch.pow(Theta, -1)
        Theta_inverse[Theta_inverse == float("Inf")] = 0

        Theta_I = (
            torch.diag(torch.pow(D_v_diag, -0.5))
            @ H
            @ torch.diag(W_e_diag + torch.ones_like(W_e_diag))
            @ torch.diag(torch.pow(D_e_diag, -1))
            @ torch.transpose(H, 0, 1)
            @ torch.diag(torch.pow(D_v_diag, -0.5))
        )

        Theta_I[Theta_I != Theta_I] = 0
        Theta_I_inverse = torch.pow(Theta_I, -1)
        Theta_I_inverse[Theta_I_inverse == float("Inf")] = 0

        Laplacian = torch.eye(Theta.shape[0]) - Theta

        fourier_e, fourier_v = torch.linalg.eigh(Laplacian)
        # fourier_e, fourier_v = np.linalg.eig(Laplacian)

        wavelets = (
            fourier_v
            @ torch.diag(torch.exp(-1.0 * fourier_e * s))
            @ torch.transpose(fourier_v, 0, 1)
        )
        wavelets_inv = (
            fourier_v
            @ torch.diag(torch.exp(fourier_e * s))
            @ torch.transpose(fourier_v, 0, 1)
        )
        wavelets_t = torch.transpose(wavelets, 0, 1)
        # 根据那篇论文的评审意见，这里用wavelets_t或许要比wavelets_inv效果更好？

        wavelets[wavelets < 0.00001] = 0
        wavelets_inv[wavelets_inv < 0.00001] = 0
        wavelets_t[wavelets_t < 0.00001] = 0
        # data.snapshots.append({"D_v_diag": D_v_diag,
        #                        "D_e_diag": D_e_diag,
        #                        "W_e_diag": W_e_diag,  # hyperedge_weight_flat
        #                        "laplacian": Laplacian,
        #                        "fourier_v": fourier_v,
        #                        "fourier_e": fourier_e,
        #                        "wavelets": wavelets,
        #                        "wavelets_inv": wavelets_inv,
        #                        "wavelets_t": wavelets_t,
        #                        "Theta": Theta,
        #                        "Theta_inv": Theta_inverse,
        #                        "Theta_I": Theta_I,
        #                        "Theta_I_inv": Theta_I_inverse,
        #                        })
        data.snapshots.append(
            {
                "Theta": torch.FloatTensor(Theta),
                "wavelets": torch.FloatTensor(wavelets),
                "wavelets_inv": torch.FloatTensor(wavelets_inv),
            }
        )
    return data


def GenerateNegHe(data):
    edge_index = data.edge_index.numpy()
    cidx = edge_index[1].min()
    edge_index[1] -= cidx

    num_hyperedges = data.num_hyperedges
    num_nodes = data.num_nodes
    num_edges = data.num_edges

    p = edge_index.shape[1] / num_hyperedges / num_nodes

    pos_H = torch.zeros((num_hyperedges, num_nodes))
    pos_H[edge_index[1], edge_index[0]] = 1
    H_dtype = pos_H.dtype
    neg_H = torch.zeros((num_hyperedges, num_nodes)).to(H_dtype)

    neg_he_idx = 0

    while neg_he_idx < num_hyperedges:
        he = np.random.choice([1, 0], size=num_nodes, p=[p, 1 - p])
        he = torch.from_numpy(he).to(H_dtype)
        if (
            not torch.any(torch.all(pos_H == he, dim=1)).item()
        ) and he.sum().item() > 0:
            neg_H[neg_he_idx] = he
            neg_he_idx += 1
    H = torch.vstack((pos_H, neg_H))
    he_idx, node_idx = H.nonzero(as_tuple=True)
    edge_index = torch.stack((node_idx, he_idx))
    y = torch.hstack((torch.ones(num_hyperedges), torch.zeros(num_hyperedges)))

    assert edge_index[0].max() + 1 == num_nodes, "The number of nodes changes"
    assert (
        edge_index[1].max() + 1 == num_hyperedges + num_hyperedges
    ), "The number of hyperedge does not match"

    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx]
    # y = y[sorted_idx]

    data.edge_index_bak = edge_index
    edge_index[1] += cidx

    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)

    data.num_edges = edge_index.shape[1]
    data.edge_index = edge_index
    data.he_y = y.type(torch.FloatTensor)
    data.num_hyperedges = num_hyperedges * 2

    return data


def GenerateNegEdges(data):
    edge_index = data.edge_index.numpy()
    cidx = edge_index[1].min()
    edge_index[1] -= cidx

    num_hyperedges = data.num_hyperedges
    num_nodes = data.num_nodes
    num_edges = data.num_edges

    class_hyperedges = data.homo_class.detach().cpu().numpy()
    y = data.y.detach().cpu().numpy()

    num_classes = data.num_classes

    assert (
        edge_index[1].max() - edge_index[1].min()
    ) + 1 == num_hyperedges, "num_hyperedges does not match!"
    # assert (edge_index[0].max() - edge_index[0].min()) + 1 == num_nodes, "num_nodes does not match!"
    assert edge_index.shape[1] == num_edges, "num_edges does not match!"

    H = np.zeros((num_hyperedges, num_nodes))
    yH = np.zeros((num_classes, num_nodes))

    for idx, label in enumerate(y):
        yH[label, idx] = 1
    for i, j in zip(edge_index[0], edge_index[1]):
        H[j, i] = 1

    neg_edges = []
    for he_idx, he in enumerate(H):
        exist_nodes = he.nonzero()[0]
        major_class = class_hyperedges[he_idx]
        candidate = (
            np.ones(num_nodes) - np.max((he, yH[major_class, :]), axis=0)
        ).nonzero()[0]
        # candidate = [node_idx for node_idx in range(num_nodes)
        #              if ((node_idx not in exist_nodes) and (y[node_idx] != major_class))]
        if len(exist_nodes) < len(candidate):
            neg_nodes = np.random.choice(
                candidate, size=len(exist_nodes), replace=False
            )
        else:
            neg_nodes = candidate
        neg_edges.extend([[node_idx, he_idx] for node_idx in neg_nodes])
    edge_index = np.concatenate([edge_index, np.array(neg_edges).T], axis=1)
    edge_index = torch.LongTensor(edge_index)
    edge_y = torch.cat((torch.ones(num_edges), torch.zeros(len(neg_edges))), dim=-1)

    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx]
    edge_y = edge_y[sorted_idx]

    data.num_edges = edge_index.shape[1]
    edge_index[1] += cidx

    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_y = torch.cat((edge_y, edge_y), dim=-1)
    data.edge_index = edge_index
    data.edge_y = edge_y
    return data


def preprocess_data(args, dataset):
    # print(f"Dataset: {args.dname}")
    data = dataset.data
    if args.model in ["HyperGCN"]:
        data = ExtractV2E(data)
    elif args.model in ["HyperGCN_HP"]:
        data = ExtractV2E(data)
        data = ConstructH(data, sparse=False, override=True)
    elif args.model in ["WCEGNN"]:
        data = ExtractV2E(data)
        data = ConstructH(data)
        # data.edge_index = torch.FloatTensor(data.edge_index)
        data.edge_index = ndarry_to_torch_sparse_tensor(data.edge_index)
        file_path = osp.join(
            args.root_dir,
            args.preprocessed_dir,
            f"Dist_{args.dname}_{args.Distance_Method}.txt",
        )
        data = GenerateDist(data, file_path, TYPE=args.Distance_Method)
    elif args.model in ["CEGCN", "CEGAT", "CEGIN"]:
        data = ExtractV2E(data)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE="V2V")
    elif args.model in ["LEGCN", "LEGAT", "LEGIN"]:
        data = ExtractV2E(data)
        data = ConstructLE(data, args.LE_V, args.LE_E, True)
    elif args.model in ["UniGAT", "UniGCN", "UniGCN2", "UniGIN", "UniSAGE"]:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = ConstructH(data)
        data.edge_index = sp.csr_matrix(data.edge_index)
        if args.cuda in [0, 1]:
            device = torch.device(
                "cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device("cpu")
        (row, col), value = torch_sparse.from_scipy(data.edge_index)
        V, E = row, col
        V, E = V.to(device), E.to(device)

        degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)

        degE = scatter(degV[V], E, dim=0, reduce="mean")
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[torch.isinf(degV)] = 1
        args.UniGNN_degV = degV
        args.UniGNN_degE = degE

        V, E = V.cpu(), E.cpu()
        del V
        del E
    elif args.model in ["HSetGNN"]:
        data = ExtractV2E(data)
        data = GenerateHetero(data, add_self_loop=args.add_self_loop)
        data = norm_contruction(data, option=args.normtype)
    elif args.model in ["SEGIN", "SEGCN", "SEGAT"]:
        data = ExtractV2E(data)
        data = ConstructH(data)
        data = GenerateSTAR(data)
    elif args.model in ["HCHA", "HGNN"]:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data.edge_index[1] -= data.edge_index[1].min()
    elif args.model in ["HNHN"]:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()
    elif args.model in ["HWNN"]:
        data = ExtractV2E(data)
        data = ConstructH(data, False)
        data = GenerateHetero(data, add_self_loop=args.add_self_loop)
        data = GenerateSnapshot(data)
    elif args.model in ["MLP"]:
        pass
    elif args.model in ["AdE", "AdE_HP"]:
        data = ExtractV2E(data)
        # data = ConstructH(data, sparse=not args.fast, override=True)
        # data.edge_index = ndarry_to_torch_sparse_tensor(data.edge_index)
        data = GenerateDist(data, TYPE="euclidean")
        # data.dists = 1
    elif args.model in ["CEGCN_HP"]:
        data = ExtractV2E(data)
        data = ConstructH(data, True)
        data.edge_index = ndarry_to_torch_sparse_tensor(data.edge_index)
        # data = ConstructV2V(data)
        # data = norm_contruction(data, TYPE="V2V")
    elif args.model in ["IRMM_GCN"]:
        data = ExtractV2E(data)
        data = ConstructH(data, override=True)
        data = IRMM(args, data, recompute=False)
    elif args.model in ["model", "AllSetTransformer", "AllDeepSets", "EDHNN"]:
        data = ExtractV2E(data)
        data = GenerateHomo(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
            args.num_hyperedges = data.num_hyperedges
        if args.exclude_self:
            data = expand_edge_index(data)
            args.num_hyperedges = data.num_hyperedges
        # if args.add_negative_edges:
        #     data = GenerateNegEdges(data, args)
        #     args.num_hyperedges = data.num_hyperedges
        #     args.num_edges = data.num_edges

        # data.norm = (data.y[data.edge_index[0]] == data.homo_class[data.edge_index[1] - data.edge_index[1].min()]).to(torch.int64)
        # print("")

        # H = ConstructH_HNHN(data)
        # he_idx = data.homo_he > 0.8
        # H = H[:, he_idx]
        # row, col = H.nonzero()
        # data.edge_index = torch.LongTensor([row, col])
        #
        edge_index = data.edge_index
        cidx = edge_index[1].min()
        edge_index[1] = edge_index[1] - cidx

        data.he = scatter(
            data.x[edge_index[0]], edge_index[1], dim=-2, reduce=args.aggregate
        )

        data.num_edges = data.edge_index.shape[1]
        args.num_edges = data.num_edges

        data = norm_contruction(data, option=args.normtype)
    else:
        raise ValueError("Unrecognized model name")
    dataset.data = data
    return dataset, args


def augmentation(args, dataset):
    data = dataset.data
    if args.aug == "None":
        data.edge_y = torch.ones(data.num_edges)
    else:
        data = ExtractV2E(data)
        data = GenerateHomo(data)
        data.num_ori_edge = data.edge_index.shape[1]
        if args.aug == "add_edge":
            data = GenerateNegEdges(data)
        elif args.aug == "add_he":
            data = GenerateNegHe(data)
            args.num_classes = args.MLP_hidden
        else:
            raise ValueError("Unrecognized augmentation method {}".format(args.aug))
    args.num_edges = data.num_edges
    args.num_hyperedges = data.num_hyperedges
    dataset.data = data
    return dataset, args
