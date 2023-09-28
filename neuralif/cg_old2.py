import torch
from scipy.sparse.linalg import gmres
import numpy as np

def gmres_(A, b, x_true, M=None, preconditioner=None, x_0=None, target=1e-4, max_iter=100_000):

    A = A.numpy()
    b = b.numpy()
    x_true = x_true.numpy()
    x_hat = x_0 if x_0 is not None else np.zeros_like(b)

    def callback(r_):
       
        #e_ = np.linalg.norm(np.abs(x - x_true))
        #r_ = np.linalg.norm(np.abs(b - A@x))

        #err_.append(e_)
        res_.append(r_)

        # errors.append((e_, r_))

        # residuals_list.append(r_)

    # initialize errors list which is a list of (error,residual)
    # e_ = np.linalg.norm(x_hat - x_true)
    # r_ = np.linalg.norm(b - A@x_hat)
    # errors = [(e_, r_)]
    # residuals_list = []

    err_ = []
    res_ = []
    errors = []
    # errors = []
    # residuals_list = []

    # if preconditioner then apply gmres with preconditioner
    if preconditioner==True:
        x_hat, _ = gmres(A, b, M=M, callback=callback)

        e_ = np.linalg.norm(np.abs(x_hat - x_true))
        r_ = np.linalg.norm(np.abs(b - A@x_hat))
        

        print('final error ||x_true - x_hat|| : ', e_)
        print('final residual ||b - A @ x_hat|| : ', r_)

        iterations = len(res_)
        
    # for baseline model, do not use preconditioner
    else:
        x_hat, _ = gmres(A, b, callback=callback)
        
        e_ = np.linalg.norm(np.abs(x_hat - x_true))
        r_ = np.linalg.norm(np.abs(b - A@x_hat))
        

        print('final error ||x_true - x_hat|| : ', e_)
        print('final residual ||b - A @ x_hat|| : ', r_)

        iterations = len(res_)
        
    return errors, x_hat, iterations, err_, res_


#@torch.compile()
def stopping_criterion_torch(A, rk, b):
    return torch.inner(rk, A@rk)

#@torch.compile()
def conjugate_gradient(A, b, x_true, x_0=None, target=1e-8, max_iter=100_000):
    x_hat = x_0 if x_0 is not None else torch.zeros_like(b)
    r = b - A@x_hat # residual
    p = r # search direction
    
    # Errors is a tuple of (error, residual)
    error_i = (x_hat - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
    res = stopping_criterion_torch(A, r, b)
    errors = [(torch.inner(error_i, A@error_i), res)]
    
    for _ in range(max_iter):
        if res < target:
            break
        
        a = torch.inner(r, r) / torch.inner(A@p, p) # step length
        x_hat = x_hat + a * p
        c = torch.inner(r, r) # Beta like precomputation
        r = r - a * (A@p)
        p = r + (torch.inner(r, r) / c) * p
        
        error_i = (x_hat - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
        res = stopping_criterion_torch(A, r, b)
        errors.append((torch.inner(error_i, A@error_i), res))
        
    return errors, x_hat


#@torch.compile()
def split_preconditioned_conjugate_gradient(L, A, b, x_true, x_0=None, target=1e-8, max_iter=100_000):
    x_hat = x_0 if x_0 is not None else torch.zeros_like(b)
    
    r = b - A@x_hat
    r_hat = L@r
    p = L.t()@r_hat
    
    # Errors is a tuple of (error, residual)
    error_i = (x_hat - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
    res = stopping_criterion_torch(A, r_hat, b)
    errors = [(torch.inner(error_i, A@error_i), res)]
    
    for _ in range(max_iter):
        if res < target:
            break
        
        a = torch.inner(r_hat, r_hat) / torch.inner(A@p, p) # step length
        x_hat = x_hat + a * p
        c = torch.inner(r_hat, r_hat) # Beta like precomputation
        r_hat = r_hat - a * (L@(A@p))
        p = L.t()@r_hat + (torch.inner(r_hat, r_hat) / c) * p
        
        error_i = (x_hat - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
        res = stopping_criterion_torch(A, r_hat, b)
        errors.append((torch.inner(error_i, A@error_i), res))
    
    return errors, x_hat