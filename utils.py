
import networkx as nx 
from networkx.readwrite import json_graph
import dgl
from collections import Counter, defaultdict
import json
import sys 
import numpy as np
import torch 
from tqdm import tqdm 
import random
import math 
import os 
import random

def load_data():

    user2id = json.load(open('./data/user2id.json'))
    job2id = json.load(open('./data/job2id.json'))
    
    user2id = {int(k):int(v) for k,v in user2id.items()}
    job2id = {int(k):int(v) for k,v in job2id.items()}
    id2user = {v:k for k,v in user2id.items()}
    id2job = {v:k for k,v in job2id.items()}

    n_users = len(user2id)
    n_jobs = len(job2id)

    edges_train = np.load('./data/apps-train.npy')
    edge_data = defaultdict(list)
    for edge in tqdm(edges_train, desc="Build graph"):
        u, j = edge
        uid = user2id[u]
        jid = job2id[j]

        edge_data[('user', 'uj', 'job')].append((uid, jid))
        edge_data[('job', 'ju', 'user')].append((jid, uid))

    for i in tqdm(range(n_users), desc="Add self-loop user edge"):
        edge_data['user','uu','user'].append((i,i))

    for i in tqdm(range(n_jobs), desc="Add self-loop job edge"):
        edge_data['job','jj','job'].append((i,i))

    edges_test = np.load('./data/apps-test.npy')
    for i in range(edges_test.shape[0]):
        edges_test[i][0] = user2id[edges_test[i][0]]
        edges_test[i][1] = job2id[edges_test[i][1]]

    user2job = defaultdict(list)
    for i, j in edges_test:
        user2job[i].append(j)
    neg_size=int(test.shape[0]/len(user2job))

    neg = []
    job_set = set(edges_test[:,1])
    for user in user2job:
        neg_jobs = random.sample(job_set.difference(user2job[user]), neg_size)
        for neg_job in neg_jobs:
            neg.append((user, neg_job))
    
    graph = dgl.heterograph(edge_data, num_nodes_dict={'user': len(user2id), 'job': len(job2id)})
    graph.nodes['user'].data['feats'] = torch.FloatTensor(np.load('./data/user-feats.npy'))
    graph.nodes['job'].data['feats'] = torch.FloatTensor(np.load('./data/job-feats.npy'))

    return graph, edges_test, np.array(neg)

class SAGEDataset():
    def __init__(self,dgl_G, batch_size=512, neg_size=20, swap=False):
        self.dgl_G = dgl_G
        self.batch_size = batch_size
        self.neg_size = neg_size
        self.swap = swap
        self.pairs = []
        self._construct_nx_graph()

    def _construct_nx_graph(self):
        dgl2nx_map = {}
        for ntype in tqdm(self.dgl_G.ntypes, desc='Read nodes from dgl'):
            for i in range(self.dgl_G.number_of_nodes(ntype)):
                dgl2nx_map[(ntype, i)] = len(dgl2nx_map) 

        nx2dgl_map = {v:k for k,v in dgl2nx_map.items()}

        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(list(range(len(nx2dgl_map))))

        for etype in tqdm(self.dgl_G.canonical_etypes, desc='Read edges from dgl'):
            ntype_src, _, ntype_dst = etype
            src_idxs, dst_idxs = self.dgl_G.all_edges(etype=etype)
            src_idxs = src_idxs.detach().cpu().numpy().tolist()
            dst_idxs = dst_idxs.detach().cpu().numpy().tolist()

            for src_idx, dst_idx in zip(src_idxs, dst_idxs):
                nx_graph.add_edge(dgl2nx_map[(ntype_src, src_idx)], dgl2nx_map[(ntype_dst, dst_idx)])
        
        self.nx_G = nx_graph
        self.iso_nodes = [node for node in nx.isolates(self.nx_G)]
        self.edges = np.array(self.nx_G.edges)
        if self.swap:
            self.edges = np.vstack([self.edges,self.edges[:,::-1]])
        self.n_edges = self.edges.shape[0]
        self.n_batches = int(math.ceil(self.n_edges/self.batch_size))

        self.degrees = np.array([self.nx_G.degree(node) for node in self.nx_G.nodes])   
        self.max_degree = int(np.max(self.degrees))

        self.dgl2nx = dgl2nx_map
        self.nx2dgl = nx2dgl_map

        ntype_list = [self.nx2dgl[i][0] for i in self.edges[:,0]]
        ntype_freq = dict(Counter(ntype_list))
        weights = [ntype_freq[ntype] for ntype in ntype_list]
        weights = np.array(weights)
        self.weights = weights/np.sum(weights)

    def __iter__(self):
        s = np.arange(self.edges.shape[0])
        np.random.shuffle(s)
        self.edges = self.edges[s]
        self.weights = self.weights[s]

        batch_idx = 0
        while(batch_idx<self.n_batches):
            start = batch_idx*self.batch_size
            batch_edges = self.edges[start:min(start+self.batch_size, self.n_edges)]
            batch_weights = self.weights[start:min(start+self.batch_size, self.n_edges)]
            batch_idx+=1
            yield batch_edges, batch_weights, self._samples_neg()

    def _samples_neg(self, unique=False):
        distortion = 0.75
        unique = False 

        weights = self.degrees**distortion
        prob = weights/weights.sum()
        sampled = np.random.choice(len(self.degrees), self.neg_size, p=prob, replace=~unique)

        return sampled


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, datadir='./model'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_best = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.datadir = datadir

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.is_best = True
        elif score < self.best_score + self.delta:
            self.is_best = False
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.is_best = True
            self.counter = 0
        if self.is_best and self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
            self.val_loss_min = val_loss
