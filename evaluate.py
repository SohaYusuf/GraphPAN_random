import argparse
import torch
import os
from torch_geometric.loader import DataLoader

from neuralif.geo import test
from neuralif.models import NeuralIF
from neuralif.data import FolderDataset
from neuralif.utils import device

# argument is the model to load and the dataset to evaluate on
def argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--dataset", type=str, required=False, default="Random")
    parser.add_argument("--n", type=int, required=False, default=50)
    parser.add_argument("--sparse", action="store_true", required=False, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    torch.set_default_dtype(torch.float)
    
    # Load the model
    model = NeuralIF(global_features=0, latent_size=16)
    
    # load the saved weights of the model
    if args.checkpoint == "latest":
        # list all the directories in the results folder
        d = os.listdir("./results/")
        d.sort()
        
        # find the latest checkpoint
        for i in range(len(d)):
            dir_contents = os.listdir("./results/" + d[-i-1])
            
            if "best_model.pt" in dir_contents:
                checkpoint = "./results/" + d[-i-1] + "/best_model.pt"
                break
        
        model.load_state_dict(torch.load(checkpoint), map_location=torch.device(device))
    elif args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device(device)))
    else:
        print("No checkpoint provided, using random weights")
    
    # Load the dataset
    size = 1
    if args.dataset == "Random":
        test_data = FolderDataset(f"./data/{args.dataset}/test/", args.n, size=size)
    elif args.dataset == "poisson":
        test_data = FolderDataset("./data/poisson/test/", 0, graph=True, size=size)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    testdata_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Evaluate the model
    test(model, testdata_loader, skip_cond=False, sparse=args.sparse, dataset=args.dataset)