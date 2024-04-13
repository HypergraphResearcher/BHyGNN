import numpy as np
import pandas as pd
import torch
import pickle
import os
from scipy.sparse import load_npz

import os.path as osp

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_sparse import coalesce
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

def sp_csr_to_torch_sp_mx(mx):
    mx = mx.tocoo()
    values = mx.data
    indices = np.stack((mx.row, mx.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mx.shape
    
    return torch.sparse_coo_tensor(i,v, torch.Size(shape))


def load_LE_dataset(path=None, dataset="ModelNet40", train_percent=0.025):
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(dataset))

    file_name = f'{dataset}.content'
    p2idx_features_labels = osp.join(path, dataset, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-1])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    #     labels = encode_onehot(idx_features_labels[:, -1])
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))

    print('load features')

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    file_name = f'{dataset}.edges'
    p2edges_unordered = osp.join(path, dataset, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print('load edges')

    projected_features = torch.FloatTensor(np.array(features.todense()))

    # From adjacency matrix to edge_list
    edge_index = edges.T
    #     ipdb.set_trace()
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1

    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    # ipdb.set_trace()

    # build torch data class
    data = Data(
        #             x = projected_features,
        x=torch.FloatTensor(np.array(features[:num_nodes].todense())),
        edge_index=torch.LongTensor(edge_index),
        y=labels[:num_nodes])

    # ipdb.set_trace()
    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)

    #     ipdb.set_trace()

    #     # generate train, test, val mask.
    num_nodes = num_nodes
    #     n_x = n_expanded
    num_class = len(np.unique(labels[:num_nodes].numpy()))
    val_lb = int(num_nodes * train_percent)
    percls_trn = int(round(train_percent * num_nodes / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.num_nodes = num_nodes
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = num_he

    return data


def load_citation_dataset(path='../hyperGCN/data/', dataset='cora', train_percent=0.025):
    '''
    this will read the citation dataset from HyperGCN, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    print(f'Loading hypergraph dataset from hyperGCN: {dataset}')

    # first load node features:
    with open(osp.join(path, dataset, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(osp.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(osp.join(path, dataset, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([node_list + edge_list,
                           edge_list + node_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features,
                edge_index=edge_index,
                y=labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)


    data.num_class = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes

    data.num_hyperedges = len(hypergraph)

    return data


def load_yelp_dataset(path='../data/raw_data/yelp_raw_datasets/', dataset='yelp',
                      name_dictionary_size=1000,
                      train_percent=0.025):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - latitude, longitude
        - state, in one-hot coding.
        - city, in one-hot coding.
        - name, in bag-of-words

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from {dataset}')

    # first load node features:
    # load longtitude and latitude of restaurant.
    latlong = pd.read_csv(osp.join(path, 'yelp_restaurant_latlong.csv')).values

    # city - zipcode - state integer indicator dataframe.
    loc = pd.read_csv(osp.join(path, 'yelp_restaurant_locations.csv'))
    state_int = loc.state_int.values
    city_int = loc.city_int.values

    num_nodes = loc.shape[0]
    state_1hot = np.zeros((num_nodes, state_int.max()))
    state_1hot[np.arange(num_nodes), state_int - 1] = 1

    city_1hot = np.zeros((num_nodes, city_int.max()))
    city_1hot[np.arange(num_nodes), city_int - 1] = 1

    # convert restaurant name into bag-of-words feature.
    vectorizer = CountVectorizer(max_features=name_dictionary_size, stop_words='english', strip_accents='ascii')
    res_name = pd.read_csv(osp.join(path, 'yelp_restaurant_name.csv')).values.flatten()
    name_bow = vectorizer.fit_transform(res_name).todense()

    features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

    # then load node labels:
    df_labels = pd.read_csv(osp.join(path, 'yelp_restaurant_business_stars.csv'))
    labels = df_labels.values.flatten()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Yelp restaurant review hypergraph is store in a incidence matrix.
    H = pd.read_csv(osp.join(path, 'yelp_restaurant_incidence_H.csv'))
    node_list = H.node.values - 1
    edge_list = H.he.values - 1 + num_nodes

    edge_index = np.vstack([node_list, edge_list])
    edge_index = np.hstack([edge_index, edge_index[::-1, :]])

    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features,
                edge_index=edge_index,
                y=labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)

    n_x = num_nodes
    #     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute

    data.train_percent = train_percent
    data.num_hyperedges = H.he.values.max()

    return data


def load_cornell_dataset(path='../data/raw_data/', dataset='amazon',
                         feature_noise=0.1,
                         feature_dim=None):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - add gaussian noise with sigma = nosie, mean = one hot coded label.

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from cornell: {dataset}')

    # first load node labels
    df_labels = pd.read_csv(osp.join(path, dataset, f'node-labels-{dataset}.txt'), names=['node_label'])
    num_nodes = df_labels.shape[0]
    labels = df_labels.values.flatten()

    # then create node features.
    num_classes = df_labels.values.max()
    features = np.zeros((num_nodes, num_classes))

    features[np.arange(num_nodes), labels - 1] = 1
    if feature_dim is not None:
        num_row, num_col = features.shape
        zero_col = np.zeros((num_row, feature_dim - num_col), dtype=features.dtype)
        features = np.hstack((features, zero_col))

    features = np.random.normal(features, feature_noise, features.shape)
    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Corenll datasets are stored in lines of hyperedges. Each line is the set of nodes for that edge.
    p2hyperedge_list = osp.join(path, dataset, f'hyperedges-{dataset}.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list,
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features,
                edge_index=edge_index,
                y=labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)


    # add parameters to attribute
    data.num_features = features.shape[-1]
    data.num_nodes = num_nodes
    data.num_hyperedges = he_id - num_nodes
    data.num_classes = len(np.unique(labels.numpy()))
    return data


def load_twitter_data(path):
    hyperedge_file_path = osp.join(path, "hyperedges.npz")
    H = load_npz(hyperedge_file_path).astype(np.int32).toarray()

    embed_file_path = osp.join(path, f"features.txt")
    X = np.loadtxt(embed_file_path)
    X = X[:, 1:]
    # # load labels
    label_file_path = osp.join(path, "label_role.txt")
    Y = np.loadtxt(label_file_path).astype(np.int32)
    Y = Y.argmax(1)
    # label_edge_file_path = osp.join(path, "label_edge.txt")
    # label_edge = np.loadtxt(label_edge_file_path)
    # assert X.shape[0] == H.shape[1] == Y.shape[0], "The dim does not match"

    features = torch.FloatTensor(X)
    labels = torch.LongTensor(Y)
    # edge_type = torch.LongTensor(label_edge)

    hyperedge_index, node_idx = H.nonzero()
    hyperedge_index += X.shape[0]
    row_indices = np.concatenate((node_idx, hyperedge_index))
    col_indices = np.concatenate((hyperedge_index, node_idx))
    edge_index = np.stack((row_indices, col_indices), axis=0)
    edge_index = torch.LongTensor(edge_index)

    dataset = Data(x=features,
                   edge_index=edge_index,
                   y=labels)
    # dataset.H = H
    # dataset.edge_type = edge_type.argmax(1)
    dataset.train_percent = 0.5
    dataset.num_hyperedges = H.shape[0]
    return dataset



def generate_synthetic_dataset(path='../data/raw_data/',
                              dataset="sythetic",
                              num_nodes_per_class=500,
                              num_classes=4,
                              num_hyperedges=1000,
                              degree=15,
                              alpha=5):
    assert degree > alpha
    he = np.zeros((num_hyperedges, degree))
    y =  np.repeat(np.arange(num_classes), num_nodes_per_class)
    he_majority_label = np.random.choice(num_classes, num_hyperedges)
    for idx, label in enumerate(he_majority_label):
        majority_nodes = np.nonzero(y==label)[0]
        minority_nodes = np.nonzero(y!=label)[0]
        num_majority = alpha
        num_minority = degree - alpha
        selected_majority_nodes = np.random.choice(majority_nodes, num_majority, replace=False)
        selected_minority_nodes = np.random.choice(minority_nodes, num_minority, replace=False)
        he[idx] = np.concatenate((selected_majority_nodes, selected_minority_nodes)).astype(np.int32)

    os.makedirs(osp.join(path, dataset), exist_ok=True)
    label_file_path = osp.join(path, dataset, f'node-labels-{dataset}.txt')
    np.savetxt(label_file_path, y,  fmt='%i', delimiter=',')
    he_file_path = osp.join(path, dataset, f'hyperedges-{dataset}.txt')
    np.savetxt(he_file_path, he, fmt='%i', delimiter=',')





# def load_wordnet_data(path='../data/raw_data/', dataset="wordnet"):

#     train_file_path = osp.join(path, dataset, "train_data.npz")
#     data = np.load(train_file_path, allow_pickle=True)

#     # test_file_path = osp.join(path, dataset, "test_data.npz")
#     # test_data = np.load(test_file_path, allow_pickle=True)

#     hyperedges = data["train_data"]
#     num_hyperedges = len(hyperedges)
#     nums_type = data["nums_type"]
#     num_relation_types = len(nums_type)
    
#     # Generate node embeddings
    
#     H = [sp.csr_matrix((np.ones(num_hyperedges), (hyperedges[:, i], range(num_hyperedges))), shape=(nums_type[i], num_hyperedges)) for i in range(num_relation_types)]
    
#     X = [H[i].dot(sp.vstack([H[j] for j in range(num_relation_types) if j != i]).T).astype('float') for i in range(num_relation_types)]
    
#     for i in range(num_relation_types):
#         col_max = np.array(X[i].max(0).todense()).flatten()
#         _, col_index = X[i].nonzero()
#         X[i].data /= col_max[col_index]
    
#     X = [sp_csr_to_torch_sp_mx(x) for x in X]
    
    
#     node_types = []
#     next_index = 0
#     for idx, num_node_type in enumerate(data["nums_type"]):
#         hyperedges[:, idx] += next_index
#         node_types += [idx] * num_node_type
#         next_index += num_node_type
#     node_types = np.array(node_types)    # size N list, where i-th element represents the noe type of node i.
#     labels = np.array([label[0] if len(label) == 1 else -1 for label in data["labels"]])
#     edge_index = np.array([ np.array([hyperedges[:, i], range(num_hyperedges)])  for i in range(num_relation_types)]).reshape(2, -1)
#     edge_index = torch.LongTensor(edge_index)
#     labels = torch.LongTensor(labels)
#     data = Data(x=X, edge_index=edge_index, y=labels)

#     return None


def save_data_to_pickle(data, p2root='../data/', file_name=None):
    '''
    if file name not specified, use time stamp.
    '''
    #     now = datetime.now()
    #     surfix = now.strftime('%b_%d_%Y-%H:%M')
    surfix = 'star_expansion_dataset'
    if file_name is None:
        tmp_data_name = '_'.join(['Hypergraph', surfix])
    else:
        tmp_data_name = file_name
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2he_StarExpan, 'bw') as f:
        pickle.dump(data, f)
    return p2he_StarExpan


class dataset_Hypergraph(InMemoryDataset):
    def __init__(self, root, name=None,
                 p2raw=None,
                 train_percent=0.01,
                 feature_noise=None,
                 transform=None, pre_transform=None):

        existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                            'NTU2012', 'Mushroom',
                            'coauthor_cora', 'coauthor_dblp',
                            'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                            'walmart-trips-100', 'house-committees-100',
                            'cora', 'citeseer', 'pubmed', "twitter",
                            "congress-bills", "congress-bills-100", "senate-committees", "senate-committees-100",
                            "contact-high-school", "contact-high-school-100", "contact-primary-school", "contact-primary-school-100",
                            "synthetic-100", "synthetic"]

        if name not in existing_dataset:
            raise ValueError(f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self.feature_noise = feature_noise

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root
        self.myraw_dir = osp.join(root, self.name, 'raw')
        self.myprocessed_dir = osp.join(root, self.name, 'processed')

        super(dataset_Hypergraph, self).__init__(osp.join(root, name), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def raw_dir(self):
    #     return osp.join(self.root, self.name, 'raw')

    # @property
    # def processed_dir(self):
    #     return osp.join(self.root, self.name, 'processed')
    @property
    def raw_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'{self.name}_noise_{self.feature_noise}']
        else:
            file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'data_noise_{self.feature_noise}.pt']
        else:
            file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.myraw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                print(p2f)
                print(self.p2raw)
                print(self.name)

                if self.name in ['cora', 'citeseer', 'pubmed']:
                    tmp_data = load_citation_dataset(path=self.p2raw,
                                                     dataset=self.name)

                elif self.name in ['coauthor_cora', 'coauthor_dblp']:
                    assert 'coauthorship' in self.p2raw
                    dataset_name = self.name.split('_')[-1]
                    tmp_data = load_citation_dataset(path=self.p2raw,
                                                     dataset=dataset_name,
                                                     train_percent=self._train_percent)

                elif self.name in ['amazon-reviews', 'walmart-trips', 'house-committees', "congress-bills", "senate-committees"
                                   "contact-high-school", "contact-primary-school"]:
                    if self.feature_noise is None:
                        raise ValueError(f'for cornell datasets, feature noise cannot be {self.feature_noise}')
                    tmp_data = load_cornell_dataset(path=self.p2raw,
                                                    dataset=self.name,
                                                    feature_noise=self.feature_noise)
                elif self.name in ['walmart-trips-100', 'house-committees-100', "congress-bills-100", "senate-committees-100",
                                   "contact-high-school-100", "contact-primary-school-100"]:
                    if self.feature_noise is None:
                        raise ValueError(f'for cornell datasets, feature noise cannot be {self.feature_noise}')
                    feature_dim = int(self.name.split('-')[-1])
                    tmp_name = '-'.join(self.name.split('-')[:-1])
                    tmp_data = load_cornell_dataset(path=self.p2raw,
                                                    dataset=tmp_name,
                                                    feature_dim=feature_dim,
                                                    feature_noise=self.feature_noise)
                elif self.name in ["synthetic"]:
                    if self.feature_noise is None:
                        raise ValueError(f'for cornell datasets, feature noise cannot be {self.feature_noise}')

                    generate_synthetic_dataset(path=self.p2raw,
                                               dataset=self.name)
                    tmp_data = load_cornell_dataset(path=self.p2raw,
                                                    dataset=self.name,
                                                    feature_noise=self.feature_noise)
                elif self.name in ["synthetic-100"]:
                    if self.feature_noise is None:
                        raise ValueError(f'for cornell datasets, feature noise cannot be {self.feature_noise}')
                    feature_dim = int(self.name.split('-')[-1])
                    tmp_name = '-'.join(self.name.split('-')[:-1])
                    generate_synthetic_dataset(path=self.p2raw, dataset=tmp_name)
                    tmp_data = load_cornell_dataset(path=self.p2raw,
                                                    dataset=tmp_name,
                                                    feature_dim=feature_dim,
                                                    feature_noise=self.feature_noise)

                elif self.name == 'yelp':
                    tmp_data = load_yelp_dataset(path=self.p2raw,
                                                 dataset=self.name,
                                                 train_percent=self._train_percent)

                elif self.name == "twitter":
                    tmp_data = load_twitter_data(path=self.p2raw)

                else:
                    tmp_data = load_LE_dataset(path=self.p2raw,
                                               dataset=self.name,
                                               train_percent=self._train_percent)

                _ = save_data_to_pickle(tmp_data,
                                        p2root=self.myraw_dir,
                                        file_name=self.raw_file_names[0])
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)



def load_data(args):
    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed', "twitter",
                        "congress-bills", "congress-bills-100", "senate-committees", "senate-committees-100",
                        "contact-high-school", "contact-high-school-100", "contact-primary-school", "contact-primary-school-100",
                        "synthetic-100", "synthetic"]

    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100',
                      'house-committees-100', "congress-bills", "congress-bills-100", "senate-committees", "senate-committees-100",
                      "contact-high-school", "contact-high-school-100", "contact-primary-school", "contact-primary-school-100",
                      "synthetic-100", "synthetic"]

    # cocitation_list = ['cora', 'citeseer', 'pubmed']

    fake_list = ["synthetic", "synthetic-100"]

    data_dir = osp.join(args.root_dir, "data")
    raw_data_dir = osp.join(data_dir, "raw_data")
    target_dir = osp.join(data_dir, "pyg_data", "hypergraph_dataset_updated")
    
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and ((dname in synthetic_list) or (dname in fake_list)):
            p2raw = raw_data_dir
            dataset = dataset_Hypergraph(root=target_dir,
                                         name=dname,
                                         feature_noise=f_noise,
                                         p2raw=p2raw)
        else:
            if dname in ['cora', 'citeseer', 'pubmed']:
                p2raw = osp.join(raw_data_dir, "cocitation")
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = osp.join(raw_data_dir, "coauthorship")
            elif dname in ['yelp']:
                p2raw = osp.join(raw_data_dir, "yelp")
            elif dname in ["twitter"]:
                p2raw = osp.join(raw_data_dir, "twitter")
            else:
                p2raw = raw_data_dir
            dataset = dataset_Hypergraph(root=target_dir,
                                         name=dname,
                                         p2raw=p2raw)

    data = dataset.data

    #         Shift the y label to start with 0
    args.num_classes = len(data.y.unique())
    data.y = data.y - data.y.min()
    if args.dname in ["twitter"]:
        args.num_classes = len(data.y.unique())
        # data.x[data.y == 1] = data.x[data.y == 1]
        # data.x[data.y == 3] = data.x[data.y == 3]
    data.num_classes = args.num_classes
    data.num_nodes = data.x.shape[0]
        # note that we assume the he_id is consecutive.
    data.num_hyperedges = int(torch.tensor(
            [data.edge_index[0].max() - data.num_nodes + 1]).item())
    # if not hasattr(data, "label_edge"):
    #     data.label_edge = torch.ones((int(data.num_hyperedges), 1))
    args.num_nodes = data.num_nodes
    args.num_features = dataset.num_features
    args.num_hyperedges = data.num_hyperedges
    dataset.data = data
    return dataset, args


# if __name__ == "__main__":
#     data = generate_sythetic_dataset()