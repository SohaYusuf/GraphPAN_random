import torch
from scipy.sparse.linalg import gmres
import numpy as np


#@torch.compile()
def stopping_criterion_torch(A, rk, b):
    return torch.inner(rk, A@rk)

    #return np.inner(rk, A@rk)

import torch
import torch.nn.functional as F


def gmres_(A, b, x_true, M=None, preconditioner=None, x_0=None, target=1e-5, max_iter=100_000):

    A = A.numpy()
    b = b.numpy()
    x_true = x_true.numpy()

    x_hat = x_0 if x_0 is not None else np.zeros_like(b)
    
    # def stopping_criterion(Ax, b):
    #     return np.linalg.norm(Ax - b) / np.linalg.norm(b)
    
    # Errors is a list of (error, residual)
    #error_i = np.linalg.norm(x_hat - x_true) if x_true is not None else np.zeros_like(b)
    #res = np.linalg.norm((A@x_hat) - b)
    #errors = [(error_i, res)]

    def callback(residuals):
        # Store the residuals during the GMRES iterations
        residuals_.append(np.linalg.norm(residuals))

    errors = []
    residuals_ = []
    
    if preconditioner:
           
        x_hat, _ = gmres(A, b, M=M, callback=callback, tol=target)
        
        error_i = np.linalg.norm(x_hat - x_true)
        for i in range(len(residuals_)):
            res = residuals_[i]
            errors.append((error_i, res))

    else:
        x_hat, _ = gmres(A, b, callback=callback, tol=target)
        
        error_i = np.linalg.norm(x_hat - x_true)
        for i in range(len(residuals_)):
            res = residuals_[i]
            errors.append((error_i, res))
        
    return errors, x_hat


# def gmres_(A, b, x_true, M=None, preconditioner=None):

#     A = A.numpy()
#     b = b.numpy()
#     x_true = x_true.numpy()

#     def stopping_criterion_np(A, rk, b):
#         return np.linalg.norm(rk)

#     def store_ilu_prec_residual(residuals):
        
#         # print(x)
#         # x = x.reshape(-1,)
        

#         # print('Type of A: ', type(A))
#         # print('Shape of A: ', A.shape)
#         # print('Type of b: ', type(b))
#         # print('Shape of b: ', b.shape)
#         # print('Type of x: ', type(x))
#         # print('Shape of x: ', x.shape)

#         error_i = x_true
#         #r_ = b - A@x
#         r_ = residuals
#         # Store the residuals during the GMRES iterations
#         res = stopping_criterion_np(A, r_, b)
#         #errors.append((np.inner(error_i, A@error_i), res))
#         errors.append((x_true, r_))

#     x_hat = np.zeros_like(b)
#     r = b - A@x_hat # residual
    
#     # error_i = (x_hat - x_true)
#     # res = stopping_criterion_np(A, r, b)
#     # errors = [(np.inner(error_i, A@error_i), res)]

#     errors = []
    
#     if preconditioner:
#         x_hat, info = gmres(A, b, M=M, callback=store_ilu_prec_residual)
#         error_i = (x_hat - x_true)
#         res = stopping_criterion_np(A, r, b)
#         #errors.append((np.inner(error_i, A@error_i), res))
#         #print('erros: ',errors)

#     else:
#         x_hat, info = gmres(A, b, callback=store_ilu_prec_residual)
#         error_i = (x_hat - x_true)
#         res = stopping_criterion_np(A, r, b)
#         #errors.append((np.inner(error_i, A@error_i), res))
#         #print('erros: ',errors)
    
#     return errors, x_hat




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