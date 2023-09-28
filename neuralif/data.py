from glob import glob

import numpy as np
import torch

from neuralif.apps.ran import *
from neuralif.utils import ddtype


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, n, graph=True, size=None) -> None:
        super().__init__()
        self.folder = folder
        if n != 0:
            if graph:
                self.files = os.listdir(folder)

                # self.files = [file for file in files if file.endswith('.pt')]
                # print(self.files)
                #self.files = list(filter(lambda x: x.split("/")[-1].split('_')[0] == str(n), glob(folder+'*.pt')))
            else:
                self.files = list(filter(lambda x: x.split("/")[-1].split('_')[0] == str(n), glob(folder+'*.npz')))
        else:
            file_ending = "pt" if graph else "npz"
            self.files = list(glob(folder+f'*.{file_ending}'))
        
        self.graph = graph
        
        if size is not None:
            assert len(self.files) >= size, f"Only {len(self.files)} files found in {folder} with n={n}"
            self.files = self.files[:size]
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"No files found in {folder} with n={n}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.graph:
            g = torch.load(os.path.join(self.folder, self.files[idx]))
            #g = torch.load(self.files[idx])
            
            # change dtype to double
            g.x = g.x.to(ddtype)
            if hasattr(g, "s"):
                g.s = g.s.to(ddtype)

            g.edge_attr = g.edge_attr.to(ddtype)

            # x_ = g.x
            # A_ = torch.sparse_coo_tensor(g.edge_index, g.edge_attr)
            # b_ = 
            
            return g
        else:
            d = np.load(self.files[idx], allow_pickle=True)
            matrix_to_graph(d["A"], d["b"])