import os
import numpy as np
from scipy.sparse import coo_matrix, diags
import torch
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.transforms import AddSelfLoops
from torch_sparse import SparseTensor

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path, unknowns, cfl):
    
    
    print('##############################################################')
    print('n: ',unknowns)
    
    # Get the directory of the current file
    current_dir = f'C:/Users/soha9/Documents/python/neuralif/neuralif_advection/data/{path}/'
    
    # Load the data
    A_matrix_path = os.path.join(current_dir, f'A_ex9_{unknowns}_{cfl}.txt')
    u_vector_path = os.path.join(current_dir, f'u_ex9_{unknowns}_{cfl}.txt')

    A_info = np.loadtxt(A_matrix_path)
    u = (np.reshape(np.loadtxt(u_vector_path), (-1,1))).astype(np.float64)

    # Extract the row, column, and value arrays from the data
    row_A = (A_info[:, 0].astype(int)) - 1
    col_A = (A_info[:, 1].astype(int)) - 1
    val_A = A_info[:, 2]
    
    # Create the sparse matrix using COO format
    A = coo_matrix((val_A, (row_A, col_A)))
    num_nodes = A.shape[0]
    
#     print(A.shape)
#     print(u.shape)

    #u = np.random.rand(num_nodes, 1).astype(np.float32)
    b = A.dot(u)

    A_coo_precon = A
    b_precon = b
    u_precon = u
    
    # calculate the inverse of the square root of D
    D = (A.diagonal()).astype(np.float64)
    D_inv_sqrt = diags(1 / np.sqrt(D), dtype=np.float64)

    # pre-multiply A and b by D_inv_sqrt using the dot method
    A_normalized = coo_matrix(D_inv_sqrt.dot(A))
    b_normalized = (D_inv_sqrt.dot(b))
    
    A = A_normalized
    b = b_normalized

    # Convert A to PyTorch SparseTensor
    indices = torch.from_numpy(np.vstack((A.row, A.col))).long()
    values = torch.from_numpy(A.data)
    shape = torch.Size(A.shape)

    A = torch.sparse.FloatTensor(indices, values, shape)
    u = torch.reshape(torch.from_numpy(u), [-1,1])
    b = torch.reshape(torch.from_numpy(b), [-1,1])

    A = torch.tensor(A, dtype=torch.float64)
    u = torch.tensor(u, dtype=torch.float64)
    b = torch.tensor(b, dtype = torch.float64)
    
    print('A shape: ', A.shape)
    print('b shape: ', b.shape)
    print('u shape: ', u.shape)
    
    residual = torch.sum(torch.abs(torch.matmul(A, u) - b))
    print(f'Residual error |Ax-b|: {residual}')
    
    return A, u, b, A_normalized, b_normalized