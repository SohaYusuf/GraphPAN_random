import torch
from scipy.sparse.linalg import gmres
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix


def gmres_with_preconditioner(A, b, u_true, tol, plot, M, method, path):

    n = A.shape[0]
    print('type A: ', type(A))

    # csr_data = A.values().numpy()
    # csr_indices = A.col_indices().numpy()
    # csr_indptr = A.crow_indices().numpy()
    # A = csr_matrix((csr_data, csr_indices, csr_indptr), shape=A.shape)
    # b = b.numpy()
    # u_true = u_true.numpy()

    # indices = A.indices().numpy()
    # values = A.indices().numpy()
    # A = coo_matrix((values, (indices[0], indices[1])), shape=A.shape)

    A = A.to_dense().numpy()
    b = b.to_dense().numpy()
    u_true = u_true.to_dense().numpy()
    
    global iteration
    iteration = 0
    residuals = []
    
    def callback(residual):
        global iteration
        iteration = iteration +1
        residual_norm = np.linalg.norm(residual)
        print(f'Iteration: {iteration} ==========> Residual: {residual_norm}')
        residuals.append(residual_norm)

    u_gmres, info = gmres(A, b, M=M, tol=tol, callback=callback, maxiter=n)

    u_gmres = u_gmres.reshape(-1,1)
    u_true = u_true.reshape(-1,1)

    error = np.linalg.norm(u_gmres - u_true)
    print('error |x_true - x_hat|: ',error)
    iterations_ = iteration

    if plot:
        print(f"Number of iterations for {method}:", iteration)
        plt.figure(1)
        plt.plot(residuals, label=method)
        plt.title(f'GMRES for random non-symmetric data (n={n})')
        plt.xlabel('# iteration')
        plt.ylabel('residual error')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{path}/{method}_gmres.png')
    
    return iterations_, residuals

def gmres_without_preconditioner(A, b, u_true, tol, plot, method, path):

    n = A.shape[0]
    # print('type A: ', type(A))
    # csr_data = A.values().numpy()
    # csr_indices = A.col_indices().numpy()
    # csr_indptr = A.crow_indices().numpy()
    # A = csr_matrix((csr_data, csr_indices, csr_indptr), shape=A.shape)
    # b = b.numpy()
    # u_true = u_true.numpy()
    
    # indices = A.indices().numpy()
    # values = A.indices().numpy()
    # A = coo_matrix((values, (indices[0], indices[1])), shape=A.shape)
    
    A = A.to_dense().numpy()
    b = b.to_dense().numpy()
    u_true = u_true.to_dense().numpy()
    
    global iteration
    iteration = 0
    residuals = []
    
    def callback(residual):
        global iteration
        iteration = iteration +1
        residual_norm = np.linalg.norm(residual)
        print(f'Iteration: {iteration} ==========> Residual: {residual_norm}')
        residuals.append(residual_norm)
            
    u_gmres, info = gmres(A, b, tol=tol, callback=callback, maxiter=n)
    u_gmres = u_gmres.reshape(-1,1)
    u_true = u_true.reshape(-1,1)

    error = np.linalg.norm(u_gmres - u_true)
    print('error |x_true - x_hat|: ',error)
    iterations_ = iteration
    
    if plot:
        print(f"Number of iterations for {method}:", iteration)
        plt.figure(1)
        plt.plot(residuals, label=method)
        plt.title(f'GMRES for random non-symmetric data (n={n})')
        plt.xlabel('# iteration')
        plt.ylabel('residual error')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{path}/{method}_gmres.png')
    
    return iterations_, residuals