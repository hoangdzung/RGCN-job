
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

def load_data():
    edges = np.load('apps.npy')

    user2id = {v:i for i,v in enumerate(sorted(list(set(edges[:,0]))))}
    job2id = {v:i for i,v in enumerate(sorted(list(set(edges[:,1]))))}

    id2user = {v:k for k,v in user2id.items()}
    id2job = {v:k for k,v in job2id.items()}

    edge_data = defaultdict(list)
    for edge in tqdm(edges, desc="Build graph"):
        u, j = edge
        uid = user2id[u]
        jid = job2id[j]

        edge_data[('user', 'uj', 'job')].append((uid, jid))
        edge_data[('job', 'ju', 'user')].append((jid, uid))

    graph = dgl.heterograph(edge_data, num_nodes_dict={'user': len(user2id), 'job': len(job2id)})
    graph.nodes['user'].data['features'] = torch.randn(len(user2id), 512)
    graph.nodes['job'].data['features'] = torch.randn(len(job2id), 512)

    return graph 
