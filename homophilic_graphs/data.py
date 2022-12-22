"""
Code partially copied from 'Diffusion Improves Graph Learning' repo https://github.com/klicperajo/gdc/blob/master/data.py
"""

import os

import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize
DATA_PATH = '../../data'

import os.path as osp
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

def bin_feat(feat, bins):
  digitized = np.digitize(feat, bins)
  return digitized - digitized.min()

def load_data_airport(dataset_str, data_path, return_label=True):
  graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
  adj = nx.adjacency_matrix(graph)
  features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
  if return_label:
    label_idx = 4
    labels = features[:, label_idx]
    features = features[:, :label_idx]
    labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
    return sp.csr_matrix(adj), features, labels
  else:
    return sp.csr_matrix(adj), features
def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]

    print("number of total positive samples: ", len(pos_idx))
    print("number of total negetive samples: ", len(neg_idx))
    print("number of training positive samples: ", len(idx_train_pos))
    print("number of training negetive samples: ", len(idx_train_neg))
    print("number of val positive samples: ", len(idx_val_pos))
    print("number of val negetive samples: ", len(idx_val_neg))
    print("number of val positive samples: ", len(idx_test_pos))
    print("number of val negetive samples: ", len(idx_test_neg))

    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def load_synthetic_data(dataset_str, use_feats, data_path):
  object_to_idx = {}
  idx_counter = 0
  edges = []
  with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
    all_edges = f.readlines()
  for line in all_edges:
    n1, n2 = line.rstrip().split(',')
    if n1 in object_to_idx:
      i = object_to_idx[n1]
    else:
      i = idx_counter
      object_to_idx[n1] = i
      idx_counter += 1
    if n2 in object_to_idx:
      j = object_to_idx[n2]
    else:
      j = idx_counter
      object_to_idx[n2] = j
      idx_counter += 1
    edges.append((i, j))
  adj = np.zeros((len(object_to_idx), len(object_to_idx)))
  for i, j in edges:
    adj[i, j] = 1.  # comment this line for directed adjacency matrix
    adj[j, i] = 1.
  if use_feats:
    features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
  else:
    features = sp.eye(adj.shape[0])
  labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
  return sp.csr_matrix(adj), features, labels

def get_dataset(opt: dict, data_dir, use_lcc: bool = True) -> InMemoryDataset:
    ds = opt['dataset']
    path = os.path.join(data_dir, ds)

    if ds in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, ds)
        use_lcc = False
    elif ds in ['Computers', 'Photo']:
        dataset = Amazon(path, ds)
    elif ds == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    elif ds == 'CoauthorPHY':
        dataset = Coauthor(path, 'Physics')
    elif ds == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=ds, root=path,
                                         transform=T.ToSparseTensor())
        use_lcc = False  # never need to calculate the lcc with ogb datasets
    elif ds == 'airport':
        dataset = Planetoid(path, 'cora')
        adj, features, labels = load_data_airport('airport', os.path.join('./dataset', 'airport'), return_label=True)

        val_prop, test_prop = 0.15, 0.15
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=1234)
        train_mask = torch.zeros(features.shape[0], dtype=bool, )
        train_mask[idx_train] = True
        test_mask = torch.zeros(features.shape[0], dtype=bool, )
        test_mask[idx_val] = True
        val_mask = torch.zeros(features.shape[0], dtype=bool, )
        val_mask[idx_test] = True
        adj = adj.tocoo()
        row, col, edge_attr = adj.row, adj.col, adj.data
        row = torch.LongTensor(row)
        col = torch.LongTensor(col)
        edge_attr = torch.FloatTensor(edge_attr)
        labels = torch.LongTensor(labels)
        features = torch.FloatTensor(features)
        edges = torch.stack([row, col], dim=0)
        data = Data(
            x=features,
            edge_index=torch.LongTensor(edges),
            edge_attr=edge_attr,
            y=labels,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask
        )
        use_lcc = False
        dataset.data = data

    elif ds == 'disease':
        dataset = Planetoid(path, 'cora')
        adj, features, labels = load_synthetic_data('disease_nc', 1, os.path.join('./dataset', 'disease_nc'), )
        val_prop, test_prop = 0.10, 0.60

        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=1234)
        train_mask = torch.zeros(features.shape[0], dtype=bool, )
        train_mask[idx_train] = True
        test_mask = torch.zeros(features.shape[0], dtype=bool, )
        test_mask[idx_val] = True
        val_mask = torch.zeros(features.shape[0], dtype=bool, )
        val_mask[idx_test] = True
        adj = adj.tocoo()
        row, col, edge_attr = adj.row, adj.col, adj.data
        row = torch.LongTensor(row)
        col = torch.LongTensor(col)
        edge_attr = torch.FloatTensor(edge_attr)
        labels = torch.LongTensor(labels)
        features = features.toarray()
        features = torch.FloatTensor(features)
        edges = torch.stack([row, col], dim=0)
        data = Data(
            x=features,
            edge_index=torch.LongTensor(edges),
            edge_attr=edge_attr,
            y=labels,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask
        )
        use_lcc = False
        dataset.data = data
    else:
        raise Exception('Unknown dataset.')

    if use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data

    train_mask_exists = True
    try:
        dataset.data.train_mask
    except AttributeError:
        train_mask_exists = False

    if ds == 'ogbn-arxiv':
        split_idx = dataset.get_idx_split()
        ei = to_undirected(dataset.data.edge_index)
        data = Data(
            x=dataset.data.x,
            edge_index=ei,
            y=dataset.data.y,
            train_mask=split_idx['train'],
            test_mask=split_idx['test'],
            val_mask=split_idx['valid'])
        dataset.data = data
        train_mask_exists = True

    if use_lcc or not train_mask_exists:
        if ds in ['Computers', 'Photo','CoauthorCS','CoauthorPHY']:
            dataset.data = get_train_val_test_split(opt['seed'],dataset.data,
                             train_examples_per_class=20, val_examples_per_class=30,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None)
        else:
            dataset.data = set_train_val_test_split(
                opt['seed'],
                dataset.data,
                num_development=5000 if ds == "CoauthorCS" else 1500)

    return dataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    print("num of train: ",len(train_idx))
    print("num of val_idx: ", len(val_idx))
    print("num of test_idx: ", len(test_idx))

    return data

def get_train_val_test_split(seed,
                             data,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    # labels = data.y
    labels = binarize_labels(data.y.numpy())
    num_nodes = data.y.shape[0]
    random_state = np.random.RandomState(seed)
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_indices)
    data.val_mask = get_mask(val_indices)
    data.test_mask = get_mask(test_indices)

    print("num of train: ",len(train_indices))
    print("num of val_idx: ", len(val_indices))
    print("num of test_idx: ", len(test_indices))

    return data


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def binarize_labels(labels, sparse_output=False, return_classes=False):
    """Convert labels vector to a binary label matrix.
    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].
    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.
    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.
    """
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix

if __name__ == '__main__':
    # example for heterophilic datasets
    from heterophilic import get_fixed_splits

    opt = {'dataset': 'Cora', 'device': 'cpu'}
    dataset = get_dataset(opt)
    for fold in range(10):
        data = dataset[0]
        data = get_fixed_splits(data, opt['dataset'], fold)
        data = data.to(opt['device'])
