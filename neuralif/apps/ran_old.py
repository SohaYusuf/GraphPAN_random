
#################################### Generate random non-symmetrices as data ################################ 

import argparse
import os

import numpy as np
import torch
import scipy
from scipy.sparse import coo_matrix
from torch_geometric.data import Data


def generate_sparse_random(n, alpha=1e-4, random_state=0, sol=False, sym=True):
    # We add to spd matricies since the sparsity is only enforced on the cholesky decomposition
    # generare a lower trinagular matrix
    # Random state
    rng = np.random.RandomState(random_state)
    
    if n == 100_000:
        zero_prob = rng.uniform(0.999, 0.9998)
    elif n > 5000 and n <= 10_000:
        zero_prob = rng.uniform(0.999, 0.9999)
    elif n >= 1000 and n <= 5_000:
        zero_prob = rng.uniform(0.993, 0.9965)
    elif n == 1_000 or n == 2_000:
        zero_prob = rng.uniform(0.98, 0.999)
    elif n == 100:
        zero_prob = rng.uniform(0.96, 0.99)
    elif n <= 50:
        zero_prob = 0.999
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
    print('number of values: ', vals.shape)
    M = coo_matrix((vals, (rows, cols)), shape=(n, n))
    I = scipy.sparse.identity(n)
    
    if sym:
        # create spd matrix
        A = M @ M.T + alpha * I               # change to A = M + alpha*I for non-symmetric problems (added by soha)
    else:
        A = M + I
    
    b = rng.uniform(0, 1, size=n)
    x = scipy.sparse.linalg.spsolve(A, b)
    #tmp = A.toarray()

    # print('A shape: ', A.shape)
    # print('b shape: ', b.shape)
    # print('x shape: ', x)

    # residual_ = np.linalg.norm(np.abs(b-tmp@x))
    # print('residual for data: ', residual_)

    # if sol:
    #     x = x
    #     #x = scipy.sparse.linalg.spsolve(A, b)
    #     return A, x, b
    # else:
    #     x = None

    return A, x, b


def matrix_to_graph_sparse(A, x_true, b):
    n = A.shape[0]
    edge_index = torch.tensor(list(map(lambda x: [x[0], x[1]], zip(A.row, A.col))), dtype=torch.long)
    edge_features = torch.tensor(list(map(lambda x: [x], A.data)), dtype=torch.float64)
    node_features = torch.tensor(list(map(lambda x: [x], b)), dtype=torch.float64)

    data = Data(x=node_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
    # data.s = torch.tensor(x_true, dtype=torch.float64)
    # data.n = n

    # convert the graph back to matrix and compute the residual error
    # graph = data
    # A_t = torch.sparse_coo_tensor(graph.edge_index, graph.edge_attr.squeeze(), requires_grad=False)
    # b_t = ((graph.x[:, 0]).squeeze()).reshape(-1,1)
    # x_t = (graph.s).reshape(-1,1)

    # A_tt = A_t.to_dense().numpy()
    # Aa = A.toarray()
    # are_equal = np.array_equal(A_tt, Aa)
    # print("Are the COO tensor and COO matrix equal?", are_equal)

    # res_error = torch.linalg.norm(b_t - torch.sparse.mm(A_t, x_t))
    # print('Residual error 2 ||b - A @ x_true||: ',res_error)

    return data


def matrix_to_graph(A, x, b):
    return matrix_to_graph_sparse(coo_matrix(A), x, b)


def create_dataset(n, samples, alpha=1e-2, graph=True, rs=0, mode='train', sym=True):
    if mode != 'train':
        assert rs != 0, 'rs must be set for test and val to avoid overlap'
    
    for sam in range(samples):
        # generate solution only for val and test
        if alpha is None:
            alpha = np.random.uniform(1e-4, 1e-2)
        
        A, x, b = generate_sparse_random(n, random_state=(rs + sam), alpha=alpha, sol=(mode == 'test'), sym=sym)
        #A, x, b = generate_sparse_random(n, random_state=(rs + sam), alpha=alpha, sol=True, sym=sym)
        #base_directory = r'C:/Users/soha9/Documents/python/neuralif/neuralif_advection/data'

        print('A shape: ', A.shape)
        print('b shape: ', b.shape)
        print('x shape: ', x.shape)

        # residual_ = np.linalg.norm(np.abs(b-A@x))
        # print('Residual error 1 ||b - A @ x_true||: ',residual_)

        if graph:
            graph = matrix_to_graph(A, x, b)
            graph.n = n
            graph.s = torch.tensor(x, dtype=torch.float64)
            # n = graph.n
            #graph.s = torch.tensor(x, dtype=torch.float)
            # if x is not None:
            #     graph.s = torch.tensor(x, dtype=torch.float)
            #graph.n = n
            torch.save(graph, f'./data/Random_{n}/{mode}/{n}_{sam}.pt')

            
        else:
            A = coo_matrix(A)
            np.savez(f'./data/Random_{n}/{mode}/{n}_{sam}.npz', A=A, b=b, x=x)

        


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--graph", action='store_true', default=False)
    parser.add_argument("--sym", type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    # script arguments
    args = argparser()
    n = args.n
    sym = args.sym
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