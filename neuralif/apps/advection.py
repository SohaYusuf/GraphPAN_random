import argparse
import os

import numpy as np
import torch
import scipy
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from load_data import *

def matrix_to_graph_sparse(A, b):
    edge_index = torch.tensor(list(map(lambda x: [x[0], x[1]], zip(A.row, A.col))), dtype=torch.long)
    edge_features = torch.tensor(list(map(lambda x: [x], A.data)), dtype=torch.float)
    node_features = torch.tensor(list(map(lambda x: [x], b)), dtype=torch.float)
    
    # Embed the information into data object
    data = Data(x=node_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
    return data

def generate_data(n, cfl, mode):
    
    A_tensor, u_tensor, b_tensor, A_coo, b_coo = load_data(f'{mode}/cfl={cfl}', n, cfl)
    
    sam = 0
    graph = matrix_to_graph_sparse(A_coo, b_coo)
    if u_tensor is not None:
        graph.s = torch.tensor(u_tensor, dtype=torch.float)
    graph.n = n
    
    path = f'C:/Users/soha9/Documents/python/neuralif/neuralif_advection/data/{mode}/cfl={cfl}/pt/{n}_{sam}.pt'
    torch.save(graph, path)
    
    # Verify .pt file is written successfully
    ddtype = torch.float16
    g = torch.load(path)
    # change dtype to double
    g.x = g.x.to(ddtype)
    if hasattr(g, "s"):
        g.s = g.s.to(ddtype)
    g.edge_attr = g.edge_attr.to(ddtype)
    
    print('Path where file is generated:', path)
    print(f'.pt file with following data is successfully generated!')
    print(g)