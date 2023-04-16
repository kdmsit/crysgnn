from __future__ import print_function, division
import scipy.sparse as sp
import csv
import functools
import json
import os
import random
import warnings
import copy
import numpy as np
import torch
import pickle as pkl
import networkx as nx
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# def get_train_val_test_loader(dataset,collate_fn=default_collate,batch_size=64,num_workers=1, pin_memory=False, **kwargs):
#     train_size = kwargs['train_size']
#     test_size = kwargs['test_size']
#
#     train_sampler = SubsetRandomSampler(train_size)
#     test_sampler = SubsetRandomSampler(test_size)
#
#     train_loader = DataLoader(dataset, batch_size=batch_size,
#                               sampler=train_sampler,
#                               num_workers=num_workers,
#                               collate_fn=collate_fn, pin_memory=pin_memory)
#     test_loader = DataLoader(dataset, batch_size=batch_size,
#                             sampler=test_sampler,
#                             num_workers=num_workers,
#                             collate_fn=collate_fn, pin_memory=pin_memory)
#     return train_loader, test_loader

def get_train_val_test_loader(dataset,total_size, collate_fn=default_collate,batch_size=64, train_ratio=None,val_ratio=0.1,
                              test_ratio=0.1, return_test=False,num_workers=1, pin_memory=False, **kwargs):
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[-(valid_size + test_size):-test_size])
    test_sampler = SubsetRandomSampler(indices[-test_size:])

    train_loader = DataLoader(dataset, batch_size=batch_size,sampler=train_sampler,num_workers=num_workers,collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,sampler=val_sampler,num_workers=num_workers,collate_fn=collate_fn, pin_memory=pin_memory)
    test_loader = DataLoader(dataset, batch_size=batch_size,sampler=test_sampler,num_workers=num_workers,collate_fn=collate_fn, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader





def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        batch_target.append(target)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


# def collate_pool(dataset_list):
#     batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
#     crystal_atom_idx, batch_target = [], []
#     batch_sg,batch_formula = [],[]
#     base_idx = 0
#     for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, spacegroup,formula) in enumerate(dataset_list):
#         n_i = atom_fea.shape[0]
#         batch_atom_fea.append(atom_fea)
#         batch_nbr_fea.append(nbr_fea)
#         batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
#         batch_target.append(target)
#         new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
#         crystal_atom_idx.append(new_idx)
#         batch_sg.append(spacegroup)
#         batch_formula.append(formula)
#         base_idx += n_i
#     return (torch.cat(batch_atom_fea, dim=0),
#             torch.cat(batch_nbr_fea, dim=0),
#             torch.cat(batch_nbr_fea_idx, dim=0),
#             crystal_atom_idx),\
#         torch.stack(batch_target, dim=0),\
#         batch_sg, batch_formula

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class PKLData(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        # Read Crystal Data
        file_path = os.path.join(self.root_dir, str(idx))
        with open(file_path, 'rb') as handle:
            sentences = pkl.load(handle)
        atom_fea, nbr_fea, nbr_fea_idx, target,spacegroup = \
            sentences[0], sentences[1], sentences[2], sentences[3], sentences[4]
        atom_fea = np.asarray(atom_fea.todense())
        nbr_fea = nbr_fea
        nbr_fea_idx = nbr_fea_idx

        atom_fea = torch.Tensor(atom_fea)
        target = torch.Tensor(target)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return (atom_fea, nbr_fea, nbr_fea_idx), target, spacegroup


class CIFData(Dataset):
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            # For Jarvis
            headings = next(reader)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join('../../atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    # @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        # # Target for Jarvis
        target = []
        data_row = copy.deepcopy(self.id_prop_data[idx])
        cif_id = str(int(float(data_row.pop(0))))
        x = data_row[0]
        target.append(float(x))

        # # Crystal
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + '.cif'))


        # Atom feature
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])

        # Neighbours
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +[self.radius + 1.] * (self.max_num_nbr -len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = sp.csc_matrix(atom_fea)
        atom_fea = np.asarray(atom_fea.todense())

        atom_fea = torch.Tensor(atom_fea)
        target = torch.Tensor(target)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
