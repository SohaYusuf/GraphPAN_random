import torch
from scipy.sparse import diags
from scipy.sparse import linalg as sla
import numpy as np
from scipy.sparse.linalg import spilu, LinearOperator, gmres


def jacobian_preconditioner(A, sparse=False):
    # We choose L = D^(1/2) = diag(a11, a22, ..., ann)
    if sparse:
        L = torch.sparse_coo_tensor(torch.arange(A.shape[0]), 1 / torch.sqrt(A.diagonal()), size=A.shape)
    else:
        L = torch.diag(1 / torch.sqrt(A.diagonal()))
    
    return L


def ichol_(l_matrix, fill_factor=1., drop_tol=0.):
    # https://scicomp.stackexchange.com/questions/7837/sparse-incomplete-cholesky
    
    lu = sla.spilu(l_matrix.tocsc(), fill_factor=fill_factor, drop_tol=drop_tol, permc_spec="NATURAL")
    L = lu.L
    D = diags(np.sqrt(lu.U.diagonal()))
    
    return L@D

def ilu_(l_matrix, fill_factor=1., drop_tol=0.):
    # https://scicomp.stackexchange.com/questions/7837/sparse-incomplete-cholesky


    #ilu = spilu(l_matrix, fill_factor=fill_factor, drop_tol=drop_tol, permc_spec="NATURAL")
    ilu = spilu(l_matrix)
    L = ilu.L
    U = ilu.U
    #D = diags(np.sqrt(lu.U.diagonal()))
    M = ILUOperator(ilu)

    return L, U, M

def ML_ilu_(l_matrix,u_matrix):
    
    ilu = ML_ILU(L=l_matrix, U=u_matrix)
    M = ILUOperator(ilu)

    return M

class ILUOperator(LinearOperator):
    def __init__(self, ilu):
        self.ilu = ilu
        self.shape = ilu.shape
        #self.dtype = ilu.dtype

    def matvec(self, x):
        return self.ilu.solve(x)

class ML_ILU:
    def __init__(self, L, U):
        self.L = L.to("cpu").numpy()
        self.U = U.to("cpu").numpy()
        self.shape = L.shape

    def solve(self, b):
        # Perform ILU solve based on L and U matrices
        y = np.linalg.solve(self.L, b)
        x = np.linalg.solve(self.U, y)
        return x