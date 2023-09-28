import argparse
import os

import numpy as np
import torch
import scipy
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from torch_geometric.data import Data


def check_generate_random(A, u, b):
    tmp = A.toarray()
    print('A shape: ', A.shape)
    print('b shape: ', b.shape)
    print('x shape: ', u.shape)
    residual_ = np.linalg.norm(np.abs(b.reshape(-1,1) - tmp @ u.reshape(-1,1)))
    print('residual for 1 generate_sparse_random: ', residual_)
    del tmp, residual_


def check_matrix_to_graph(data):
    A_t = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(), requires_grad=False)
    b_t = ((data.x[:, 0]).squeeze()).reshape(-1,1)
    x_t = (data.s).reshape(-1,1)
    res_error = torch.linalg.norm(b_t - torch.sparse.mm(A_t, x_t))
    print('Residual error 2 for matrix_to_graph_sparse: ',res_error)
    del A_t, b_t, x_t, res_error


def generate_sparse_random(n, alpha=1e-4, random_state=0, sol=False):
    # We add to spd matricies since the sparsity is only enforced on the cholesky decomposition
    # generare a lower trinagular matrix
    # Random state
    rng = np.random.RandomState(random_state)
    
    if n == 100_000:
        zero_prob = rng.uniform(0.999, 0.9998)
    elif n > 5000 and n <= 10_000:
        zero_prob = rng.uniform(0.995, 0.998)
    elif n >= 1000 and n <= 5_000:
        zero_prob = rng.uniform(0.993, 0.9965)
    elif n == 1_000 or n == 2_000:
        zero_prob = rng.uniform(0.98, 0.999)
    elif n == 100:
        zero_prob = rng.uniform(0.96, 0.99)
    elif n <= 50:
        zero_prob = 0.9
    else:
        raise NotImplementedError(f"Can\'t generate sparse matrix for n={n}")
    
    # old code:
    # S = rng.binomial(1, (1 - zero_prob), size=(n, n))
    # M = rng.normal(0, 1, size=(n, n))
    # M = S * M # enforce sparsity
    
    nnz = int((1 - zero_prob) * n ** 2)
    rows = [rng.randint(0, n) for _ in range(nnz)]
    cols = [rng.randint(0, n) for _ in range(nnz)]
    
    uniques = set(zip(rows, cols))
    rows, cols = zip(*uniques)
    
    # generate values
    vals = np.array([rng.normal(0, 1) for _ in cols])
    M = coo_matrix((vals, (rows, cols)), shape=(n, n))
    I = scipy.sparse.identity(n)
    
    if sym:
        # create spd matrix
        A = M @ M.T + alpha * I               # change to A = M + alpha*I for non-symmetric problems (added by soha)
    else:
        A = M + I
    
    b = rng.uniform(0, 1, size=n)
    
    if check:
        x = scipy.sparse.linalg.spsolve(A, b)
        check_generate_random(A, x, b)

    else:
        if sol:
            x = scipy.sparse.linalg.spsolve(A, b)
            return A, x, b
        else:
            x = None
    
    return A, x, b


def matrix_to_graph_sparse(A, b):
    edge_index = torch.tensor(list(map(lambda x: [x[0], x[1]], zip(A.row, A.col))), dtype=torch.long)
    edge_features = torch.tensor(list(map(lambda x: [x], A.data)), dtype=torch.float64)
    node_features = torch.tensor(list(map(lambda x: [x], b)), dtype=torch.float64)
    
    # Embed the information into data object
    data = Data(x=node_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
    return data


def matrix_to_graph(A, b):
    return matrix_to_graph_sparse(coo_matrix(A), b)


def create_dataset(n, samples, alpha=1e-2, graph=True, rs=0, mode='train'):
    if mode != 'train':
        assert rs != 0, 'rs must be set for test and val to avoid overlap'
    
    i = 0
    for sam in range(samples):
        # generate solution only for val and test
        if alpha is None:
            alpha = np.random.uniform(1e-4, 1e-2)
        
        A, x, b = generate_sparse_random(n, random_state=(rs + sam), alpha=alpha, sol=(mode == 'test'))

        print(f"------------- Sample: {i} ------------")

        if check:
            check_generate_random(A, x, b)
            print('shape A: ', A.shape)
            print('shape b: ', b.shape)
            print('shape x: ', x.shape)

        if graph:
            graph = matrix_to_graph(A, b)
            if x is not None:
                graph.s = torch.tensor(x, dtype=torch.float64)
            graph.n = n
            if check:
                check_matrix_to_graph(graph)
            torch.save(graph, f'./data/Random_{n}/{mode}/{n}_{sam}.pt')
        else:
            A = coo_matrix(A)
            np.savez(f'./data/Random_{n}/{mode}/{n}_{sam}.npz', A=A, b=b, x=x)
        i = i+1


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--graph", action='store_true', default=False)
    parser.add_argument("--check", type=int, default=0)
    parser.add_argument("--sym", type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    # script arguments
    args = argparser()
    n = args.n
    sym = args.sym
    check = args.check
    # samples = args.samples
    np.random.seed(0)
    
    # logging
    #print(f"Creating random dataset with {samples} samples for n={n}")
    
    # create the folders and subfolders where the data is stored
    os.makedirs(f'./data/Random_{n}/train', exist_ok=True)
    os.makedirs(f'./data/Random_{n}/val', exist_ok=True)
    os.makedirs(f'./data/Random_{n}/test', exist_ok=True)
    
    # create all datasets
    create_dataset(n, 5000, mode='train', rs=0, graph=True)
    create_dataset(n, 25, mode='val', rs=10000, graph=True)
    create_dataset(n, 5, mode='test', rs=103600, graph=True)