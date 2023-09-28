########## MAKE SURE ALL TENSORS ARE IN FLOAT64 ##############

import os
import time

import matplotlib.pyplot as plt
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from neuralif.cg import *
from neuralif.data import *
from neuralif.models import *
from neuralif.preconditioners import *
from neuralif.utils import *
from neuralif.paths import *

import scipy.sparse.linalg
import scipy.sparse
import torch.sparse

# Ignore pytorch warnings
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


def kA_bound(cond, k):
    return 2 * ((torch.sqrt(cond) - 1) / (torch.sqrt(cond) + 1)) ** k


def eigenval_distribution(P, A, split=True, invert=False):
    if P == None:
        return torch.linalg.eigh(A)[0]
    
    if invert:
        if split:
            P = torch.linalg.solve_triangular(P, torch.eye(P.size()[0], device=P.device, requires_grad=False), upper=False)
        else:
            P = torch.linalg.inv(P)
    
    if split:
        r = P@A@P.T
    else:
        r = P@A
    
    # eigvals = torch.linalg.eigvals(r)
    eigvals = torch.linalg.eigh(r)[0]
    return eigvals


def condition_number(P, A, invert=False, split=True):
    if invert:
        if split:
            #P = torch.linalg.solve_triangular(P, torch.eye(P.size()[0], device=P.device, requires_grad=False), upper=False)
            print(device)
            P_dense = (P.to_dense()).to(device)
            P_dense = (torch.linalg.solve_triangular(P_dense, torch.eye(P_dense.size()[0], device=P_dense.device, requires_grad=False), upper=False)).to(device)
            
            #P = (torch.sparse_coo_tensor((torch.eye(P_dense.size()[0])).to(device), P_dense.to(device))).to(device)

        else:
            P = torch.linalg.inv(P)
    
    if split:
        # P.T@A@P is wrong!
        # Not sure what the difference is between P@A@P.T and P.T@P@A?
        return torch.linalg.cond(P@A@P.T)
    else:
        return torch.linalg.cond(P@A)


def loss_chol(L, U, A, sparse=True):
    dev = L.device
    
    # * Cholesky decomposition style loss
    # if L.requires_grad:
    #     assert not sparse, "Gradient computation not supported for sparse matrices"
    
    if sparse:
        # https://github.com/pytorch/pytorch/issues/95169
        # https://github.com/rusty1s/pytorch_sparse/issues/45
        # L = L.to_sparse_csr() # convert to sparse CSR format for performance (note: does not support gradients)
        P = (torch.sparse.mm(L, U)).to(dev)
        r = (A - P).to(dev)
        return frob_norm_sparse(r.coalesce())
    else:
        A = (A.to(dev)).to_dense().squeeze()
        L = (L.to(dev)).to_dense().squeeze()
        U = (U.to(dev)).to_dense().squeeze()
        return torch.linalg.norm(L@U - A, ord="fro")


@torch.inference_mode()
def test(model, test_loader, invert=True, skip_cond=True, sparse=True, dataset="random"):
    os.makedirs(folder, exist_ok=True)
    
    model.eval()
    model = model.to(device)

    print()
    print(f"Test performance: {len(test_loader.dataset)} samples")
    print()
    
    for method in ["baseline", "ilu", "learned"]:
        print(f"Testing {method} preconditioner")
        test_results = TestResults(method, dataset)
        
        for i, data in enumerate(test_loader):
            plot = i % 10 == 0
            
            #x_true = data.s if hasattr(data, "s") else None
            # get original problem formulation
            A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(), requires_grad=False)
            A = A.to_dense()
            b = (data.x[:, 0].squeeze()).reshape(-1,1)
            x_true = (data.s).reshape(-1,1)

            # compute residual error before testing - make sure all tensors have float64 dtype
            res_error = torch.linalg.norm(torch.abs(b - torch.sparse.mm(A, x_true)))
            print('Residual error ||b - A @ x_true||: ',res_error)

            start = time.process_time_ns()
            kwargs = {"split": True}

            if method == "ilu":
                # convert A to a scipy sparse coo matrix
                A_s = A.numpy()
                A_s = scipy.sparse.coo_matrix(A_s)
                # using scipy functionality to compute incomplete LU
                L, U, M = ilu_(A_s)
                L = torch.Tensor(L.toarray())
                U = torch.Tensor(U.toarray())
                kwargs["invert"] = True
            
            elif method == "baseline":
                # ! just for convenience is not actually used...
                L = torch.eye(A.shape[0])
                U = torch.eye(A.shape[0])
                kwargs["invert"] = False
            
            elif method == "learned":
                data = data.to(device)
                L, U, _ = model(data)
                L = L.to_dense().squeeze()
                U = U.to_dense().squeeze()
                M = ML_ilu_(L,U)
                kwargs["invert"] = invert
            
            if kwargs["invert"]:
                # invert matrix if necessary efficiently...(time included in P-time!)
                I = torch.eye(L.shape[0]).to(device)
                P = torch.linalg.solve_triangular(U.to(device), torch.linalg.solve_triangular(L.to(device), I, upper=False), upper=True).to(device) # new
            else:
                P = L@U
            
            stop = time.process_time_ns()
            p_time = (stop - start) / 10e6
            print('p_time (test time): ',p_time)
            
            # make sure everything is one the same device
            A = A.to("cpu")
            data = data.to("cpu")
            P = P.to("cpu")
            L = L.to("cpu")
            U = U.to("cpu")
            
            # make A and P sparse
            if sparse:
                A = A.to_sparse_csr()
                P = P.to_sparse_csr()
            
            target = 1e-4
            start = time.process_time_ns()

            if method == "baseline":
                print("--------------------------- Baseline Errors ------------------------------------")
                iterations_, residuals = gmres_without_preconditioner(A, b, x_true, tol=target, plot=True, method='Baseline', path = f"{folder}")
                #res, x_hat, iter_, err_, res_ = gmres_(A, b, solution, preconditioner=False)
                #res, x_hat_baseline, iter_ = gmres_(A, b, solution, preconditioner=False)
                
            elif method == "ilu":
                print("--------------------------- ILU Errors ------------------------------------")
                iterations_, residuals = gmres_with_preconditioner(A, b, x_true, tol=target, plot=True, M=M, method='ILU', path = f"{folder}")
                #res, x_hat, iter_, err_, res_ = gmres_(A, b, solution, M=M, preconditioner=True)
                #res, x_hat_ilu, iter_ = gmres_(A, b, solution, M=M, preconditioner=True)
                
            elif method == "learned":
                print("--------------------------- Learned Errors ------------------------------------")
                iterations_, residuals = gmres_with_preconditioner(A, b, x_true, tol=target, plot=True, M=M, method='Learned', path = f"{folder}")
                #res, x_hat, iter_, err_, res_ = gmres_(A, b, solution, M=M, preconditioner=True)
                #res, x_hat_learned, iter_ = gmres_(A, b, solution, M=M, preconditioner=True)
                
                
            # else:
            #     res, _ = split_preconditioned_conjugate_gradient(P, A, b, solution, target=target)

        #     stop = time.process_time_ns()
            
        #     cg_time = (stop - start) / 10e6
        #     print('number of iterations: ', iterations_)
        #     print('cg_time: ', cg_time)
            
        #     res = residuals
        #     # log the cg run
        #     test_results.log_cg(cg_time, len(res) - 1, np.array([r[0].item() for r in res]), np.array([r[1].item() for r in res]), p_time)
            
        #     # form dense matrix A for analysis!
        #     if not skip_cond:
        #         A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, requires_grad=False).to_dense().squeeze()
        #         P = P.to_dense().squeeze()
            
        #         c = condition_number(P, A)
        #         loss = loss_chol(L, U, A, sparse=False)
            
        #         m = f"\n{model.__class__.__name__}" if method == "learned" else ""
            
        #         # log the results
        #         n = A.shape[0]
        #         test_results.A, test_results.L = A, L
        #         nnzA = num_non_zeros(torch.tril(A))
        #         nnzL = num_non_zeros(L)
        #         test_results.log(n, c, loss, nnzA, nnzL, plot=plot, m=m)
            
        #         dist = eigenval_distribution(P, A, split=True)
        #         test_results.log_eigenval_dist(dist, plot=True, m=m)
            
        #         # check convergence speed etc.
        #         # error_0 = res[0][0]
        #         # bounds = [error_0 * kA_bound(c, k) for k in range(len(res))]
    
        #         # plot using matplotlib
        #         if plot:
        #             #plt.plot(err_, label="error ($|| x_i - x_* ||$)")
        #             plt.plot(residuals, label="residual ($||r ||_2$)")


        #             # plt.plot([r[0] for r in res], label="error ($|| x_i - x_* ||$)")
        #             # plt.plot([r[1] for r in res], label="residual ($||r ||_2$)")
        #             # if not skip_cond:
        #             #     plt.plot(bounds, "--", label="k(A)-bound")
        #             # plt.plot([1e-8 for _ in res], ":")
                    
        #             plt.yscale("log")
        #             plt.title(f"Convergence: {method} in {iterations_} iterations" + m)
        #             plt.xlabel("iteration")
        #             plt.ylabel("log10")
        #             plt.legend()
                    
        #             plt.savefig(f"{folder}/convergence_{method}_{i}.png")
        #             plt.close()
        
        # test_results.save_results()
        # test_results.print_summary()


@torch.no_grad()
def validate(model, validation_loader, epoch, skip_cond=True,  sparse=False, **kwargs):
    # print("kwargs")
    # print(**kwargs)
    model.eval()
                
    acc_loss = 0.0
    acc_cond = 0.0
    acc_rel_cond = 0.0
    
    for data in validation_loader:
        data = data.to(device)
        
        output_L, output_U, _ = model(data)
        P = torch.sparse.mm(output_L,output_U)

        if sparse:
             A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr[:, 0])
             l = loss_chol(output_L, output_U, A, sparse=True)
        else:
            output_L = output_L.to_dense()
            output_U = output_U.to_dense()
            A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr[:, 0]).to_dense().squeeze()
            l = loss_chol(output_L, output_U, A, sparse=False)

        # output = output.to_dense()
        # A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr[:, 0]).to_dense().squeeze()
        # l = loss_chol(output, A, sparse=False)

        acc_loss += l.item()
        
        # Condition number (mean and relative)
        if not skip_cond:
            c = condition_number(P, A, **kwargs)
            acc_cond += torch.log(c)
            cA = torch.linalg.cond(A)
            acc_rel_cond += torch.log10(c / cA)
        else:
            c = torch.Tensor([0])
            acc_cond = torch.Tensor([0])
            cA = torch.Tensor([0])
            acc_rel_cond = torch.Tensor([0])
    
    print(f"Validation loss: {acc_loss / len(validation_loader)}")
    return acc_loss / len(validation_loader), acc_cond / len(validation_loader)


def main(config):
    os.makedirs(folder, exist_ok=True)
    save_dict_to_file(config, os.path.join(folder, "config.json"))
    
    # global seed-ish
    torch_geometric.seed_everything(config["seed"])
    
    # command line arguments
    args = {k: config[k] for k in ["latent_size", "num_layers", "message_passing_steps", 
                                   "decode_nodes", "encoder_layer_norm", "mp_layer_norm",
                                   "aggregate", "activation", "skip_connections", "multi_graph",
                                   "global_features", "symmetric", "sparse"]}
    
    print(args)

    sparse = config['sparse']
    
    # Create model
    model = NeuralIF(**args)
    config["invert"] = True
    model.to(device)
    
    print(f"Number params in model: {count_parameters(model)}")
    print()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # Setup datasets
    test_samples = 1
    if config["dataset"] == "random":
        # data loading
        if config['symmetric']:
            path = ran_symmetric_path()
            train_data = FolderDataset(os.path.join(path, "train"), config["n"])
            validation_data = FolderDataset(os.path.join(path, "val"), config["n"])
            test_data = FolderDataset(os.path.join(path, "test"), config["n"], size=test_samples)
        else:
            path = ran_non_symmetric_path(config['n'])
            train_data = FolderDataset(os.path.join(path, "train"), config["n"])
            validation_data = FolderDataset(os.path.join(path, "val"), config["n"])
            test_data = FolderDataset(os.path.join(path, "test"), config["n"], size=test_samples)
    
    elif config["dataset"] == "poisson":
        train_data = FolderDataset("./data/poisson/train/", 0, graph=True)
        validation_data = FolderDataset("./data/poisson/val/", 0, graph=True)
        test_data = FolderDataset("./data/poisson/test/", 0, graph=True, size=test_samples)
    
    elif config["dataset"] == "advection":
        path = mfem_advection_path()
        train_data = FolderDataset(path, 0, graph=True)
        validation_data = FolderDataset(path, 0, graph=True)
        test_data = FolderDataset(path, 0, graph=True, size=test_samples)
    
    else:
        raise NotImplementedError("Dataset not implemented, Available: random, poisson, ipm")
    
    # Data Loaders
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Initialize logging
    best_val = np.float64("inf")
    logger = TrainResults()
    
    # Train loop
    for epoch in range(config["num_epochs"]):
        running_loss = 0.0
        grad_norm = 0.0
        
        for it, data in enumerate(train_loader):
            model.train()
            data = data.to(device)
            
            optimizer.zero_grad()
            output_L, output_U, nodes = model(data)

            if sparse:
                A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr[:, 0])
            else:
               A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr[:, 0]).to_dense().squeeze() 
            
            l = loss_chol(output_L, output_U, A, sparse=sparse)
            l.backward()
            
            # check_symmetry(A)

            # track the gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            
            grad_norm += total_norm ** 0.5
            running_loss += l.item() / config["batch_size"]

            if "gradient_clipping" in config and config["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
            
            optimizer.step()
            
            # Do validation after 100 updates (to support big datasets)
            # convergence is expected to be pretty fast...
            if (it + 1) % 500 == 0:
                try:
                    val_loss, val_cond = validate(model, validation_loader, epoch, skip_cond=True, sparse=config["sparse"])
                except torch._C._LinAlgError:
                    print("WARNING: numerical instabilities, can not compute the condition number")
                    val_loss = np.float64("inf")
                    
                # use scheduler
                scheduler.step(val_loss)
                
                logger.log_val(val_loss, val_cond)
                
                if val_loss < best_val:
                    torch.save(model.to(torch.float64).state_dict(), f"{folder}/best_model.pt")
                    best_val = val_loss
        
        # log to results
        logger.log(running_loss/len(train_loader), grad_norm/len(train_loader))
        print(f"Epoch {epoch+1} loss: {1/len(train_loader) * running_loss}")
        
        if (epoch + 1) % 10 == 0:
            try:
                val_loss, val_cond = validate(model, validation_loader, epoch, invert=config["invert"], sparse=config["sparse"])
            except torch._C._LinAlgError:
                print("WARNING: numerical instabilities, can not compute the condition number")
                val_loss = np.float64("inf")
            
            logger.log_val(val_loss, val_cond)
            if val_loss < best_val:
                torch.save(model.to(torch.float64).state_dict(), f"{folder}/best_model.pt")
                best_val = val_loss
    
    # save fully trained model
    logger.save_results()
    torch.save(model.to(torch.float64).state_dict(), f"{folder}/final_model.pt")
    
    # Test the model
    print()
    print("Best validation loss:", best_val)
    
    model.load_state_dict(torch.load(f"{folder}/best_model.pt"))
    test(model, test_loader, dataset=config["dataset"])