import datetime
import json
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_sparse import SparseTensor, spadd, spspmm
import scipy.stats as st


# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Device settings
if device == "cuda":
    ddtype = torch.float64
else:
    ddtype = torch.float64
torch.set_default_dtype(ddtype)

# Folder to save the results to
folder = "results/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class ToLowerTriangular(torch_geometric.transforms.BaseTransform):
    def __init__(self, inplace=False):
        self.inplace = inplace
        
    def __call__(self, data):
        if not self.inplace:
            data = data.clone()
        
        # transform the data into lower triag graph
        # this should be a data transformation (maybe?)
        rows, cols = data.edge_index[0], data.edge_index[1]
        fil = cols <= rows
        l_index = data.edge_index[:, fil]
        edge_embedding = data.edge_attr[fil]
        
        data.edge_index, data.edge_attr = l_index, edge_embedding
        return data

class ToUpperTriangular(torch_geometric.transforms.BaseTransform):
    def __init__(self, inplace=False):
        self.inplace = inplace
        
    def __call__(self, data):
        if not self.inplace:
            data = data.clone()
        
        # transform the data into upper triangular graph
        # this should be a data transformation
        rows, cols = data.edge_index[0], data.edge_index[1]
        fil = cols >= rows
        u_index = data.edge_index[:, fil]
        edge_embedding = data.edge_attr[fil]
        
        data.edge_index, data.edge_attr = u_index, edge_embedding
        return data
        
@dataclass
class TestResults:
    method: str
    dataset: str
    
    # general parameters
    seed: int = 0
        
    # store the results of the test evaluation
    n: List[int] = field(default_factory=list)
    cond_pa: List[float] = field(default_factory=list)
    nnz_a: List[float] = field(default_factory=list)
    nnz_p: List[float] = field(default_factory=list)
    time: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    
    # store results from cg run
    cg_time: List[float] = field(default_factory=list)
    cg_iterations: List[float] = field(default_factory=list)
    cg_error: List[float] = field(default_factory=list)
    cg_residual: List[float] = field(default_factory=list)
    
    # more advanved loggings (not always set)
    distribution: List[torch.Tensor] = field(default_factory=list)
    L: torch.Tensor = None
    A: torch.Tensor = None
    
    def log(self, n, cond_pa, loss, nnz_a, nnz_p, plot=False, m=""):
        # update all arrays with the new results
        self.n.append(n)
        self.cond_pa.append(cond_pa.cpu())
        # self.time.append(time)
        self.loss.append(loss.cpu())
        self.nnz_a.append(nnz_a.cpu())
        self.nnz_p.append(nnz_p.cpu())
        
        if plot:
            i = len(self.cg_time) - 1
            
            fig, axs = plt.subplots(1, 3, figsize=plt.figaspect(1/3))
            fig.suptitle(f"{self.method.upper()} Error: {self.loss[-1]:.2f}" + m)
            
            im1 = axs[0].imshow(torch.abs(self.A), interpolation='none', cmap='Blues')
            im1.set_clim(0, 1)
            axs[0].set_title("A")
            im2 = axs[1].imshow(torch.abs(self.L), interpolation='none', cmap='Blues')
            im2.set_clim(0, 1)
            axs[1].set_title("L")
            im3 = axs[2].imshow(torch.abs(self.L@self.L.T - self.A), interpolation='none', cmap='Reds')
            im3.set_clim(0, 1)
            axs[2].set_title("L@L.T - A")
            
            # add colorbat
            fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.4, hspace=0.1)
            cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
            fig.colorbar(im3,cax=cb_ax)
            
            # share y-axis
            # for ax in fig.get_axes():
            #     ax.label_outer()
                            
            # save as file
            plt.savefig(f"{folder}/chol_factorization_{self.method}_{i}.png")
            plt.close()
    
    def log_cg(self, cg_time, cg_iterations, cg_error, cg_residual, p_time):
        self.cg_time.append(cg_time)
        self.cg_iterations.append(cg_iterations)
        self.cg_error.append(cg_error)
        self.cg_residual.append(cg_residual)
        self.time.append(p_time)
    
    def log_eigenval_dist(self, dist, plot=False, m=""):
        self.distribution.append(dist)
        
        if plot:
            i = len(self.distribution) - 1
            c = torch.max(dist) / torch.min(dist)
            
            plt.hist(dist.tolist(), density=True, bins=20, alpha=0.7)
            mn, mx = plt.xlim()
            plt.xlim(mn, mx)
            kde_xs = np.linspace(mn, mx, 300)
            kde = st.gaussian_kde(dist.tolist())
            plt.plot(kde_xs, kde.pdf(kde_xs), "--", alpha=0.7)
            plt.title(f"Eigenvalues: {self.method}, $\kappa(A)=${c.item():.2e}" + m)
            plt.ylabel("Frequency")
            plt.xlabel("$\lambda$")
            plt.savefig(f"{folder}/eigenvalues_{self.method}_{i}.png")
            plt.close()
    
    def print_summary(self):
        print(f"Method: {self.method}")
        for key, value in self.get_summary_dict().items():
            print(f"{key}: {value}")
        print()
        
    def get_summary_dict(self):
        return {
            f"cond_pa_{self.method}": np.mean(self.cond_pa),
            f"time_{self.method}": np.mean(self.time),
            f"cg_time_{self.method}": np.mean(self.cg_time),
            f"cg_iterations_{self.method}": np.mean(self.cg_iterations),
            f"total_time_{self.method}": np.mean(list(map(lambda x: x[0] + x[1], zip(self.time, self.cg_time)))),
            f"loss_{self.method}": np.mean(self.loss),
            f"nnz_a_{self.method}": np.mean(self.nnz_a),
            f"nnz_p_{self.method}": np.mean(self.nnz_p),
        }
    
    def save_results(self):
        fn = f"{folder}/test_{self.method}.npz"
        np.savez(fn, n=self.n, cond_pa=self.cond_pa, time=self.time, loss=self.loss, 
                 nnz_a=self.nnz_a, nnz_p=self.nnz_p, cg_time=self.cg_time, cg_iterations=self.cg_iterations,
                 cg_error=np.asarray(self.cg_error, dtype="object"), cg_residual=np.asarray(self.cg_residual, dtype="object"))
    
    
@dataclass
class TrainResults:
    # training
    loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    
    # validation
    log_freq: int = 10
    val_loss: List[float] = field(default_factory=list)
    val_cond: List[float] = field(default_factory=list)
    
    def log(self, loss, grad_norm):
        self.loss.append(loss)
        self.grad_norm.append(grad_norm)
        
    def log_val(self, val_loss, val_cond):
        self.val_loss.append(val_loss)
        self.val_cond.append(val_cond.cpu())
        
    def save_results(self):
        fn = f"{folder}/training.npz"
        np.savez(fn, loss=self.loss, grad_norm=self.grad_norm,
                 val_loss=self.val_loss, val_cond=self.val_cond)


def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        file.write(json.dumps(dictionary))


def invert_lower_triangular(L):
    # L is lower triangular
    # returns the inverse of L
    # L @ L_inv = I
    return torch.triangular_solve(torch.eye(L.shape[0], device=device), L, upper=False)[0]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def num_non_zeros(P):
    return torch.linalg.norm(P.flatten(), ord=0)


def sym_sparse(edge_index, edge_attr, num_nodes):
    indexA = edge_index
    indexB = torch.stack([edge_index[1], edge_index[0]], dim=0)
    
    return spadd(indexA, edge_attr, indexB, edge_attr, m=num_nodes, n=num_nodes)


def tril_sparse(src, k=0):
    row, col, value = src.coo()
    mask = row >= col - k
    new_row, new_col = row[mask], col[mask]
    new_value = value[mask]
    
    return SparseTensor(row=new_row, rowptr=None, col=new_col, value=new_value,
                        sparse_sizes=src.sparse_sizes(), is_sorted=True)
    

def squeeze_sparse(src):
    row, col, value = src.coo()
    value = value.squeeze(dim=1)
    return SparseTensor(row=row, rowptr=None, col=col, value=value,
                        sparse_sizes=src.sparse_sizes(), is_sorted=True)
    

def diag_sparse(src):
    row, col, value = src.coo()
    mask = row == col
    new_row, new_col = row[mask], col[mask]
    new_value = value[mask]
    return SparseTensor(row=new_row, rowptr=None, col=new_col, value=new_value,
                        sparse_sizes=src.sparse_sizes(), is_sorted=True)


def llt_mm(src, requires_grad=True):
    #! lacks autograd at the moment...
    # Compute the matrix multiplication L@L.T using sparse matricies
    n = src.size()[0]
    transpose_index = torch.stack([src.indices()[1], src.indices()[0]], dim=0)
    new_index, new_val = spspmm(src.indices(), src.values(), transpose_index, src.values(), n, n, n)
    new = torch.sparse_coo_tensor(new_index, new_val, (n, n), requires_grad=requires_grad)
    return new


def frob_norm_sparse(A):
    return torch.pow(torch.sum(torch.pow(A.values(), 2)), 0.5)
        

def filter_small_values(A, threshold=1e-5):
    # only keep the values above threshold
    return torch.where(torch.abs(A) < threshold, torch.zeros_like(A), A)


def plot_graph(data):
    # transofrm to networkx
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    # remove the self loops for readability
    filtered_edges = list(filter(lambda x: x[0] != x[1], g.edges()))
    nx.draw(g, edgelist=filtered_edges)
    plt.show()


def print_graph_statistics(data):
    print(data.validate())
    print(data.is_directed())
    print(data.num_nodes) 


def check_symmetry(A):
    # Check if A is symmetric
    tmp = A.to_dense()
    is_symmetric = torch.allclose(tmp, tmp.t())
    print(f"A is symmetric: {is_symmetric}")

def check_residual(A,x,b):
    result = np.dot(A.toarray(), x)  # Convert the result to a dense matrix
    residual = np.linalg.norm(np.abs(b - result))
    print(residual)
    return residual 