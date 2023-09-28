import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import aggr

from neuralif.utils import *

############################
#          Layers          #
############################
class GraphNet(nn.Module):
    # Follows roughly the outline of torch_geometric.nn.MessagePassing()
    # As shown in https://github.com/deepmind/graph_nets
    # Here is a helpful python implementation:
    # https://github.com/NVIDIA/GraphQSat/blob/main/gqsat/models.py
    # Also allows multirgaph GNN via edge_2_features 
    def __init__(self, node_features, edge_features, edge_2_features=0, global_features=0, 
                 layer_norm=False, hidden_size=0, aggregate="mean", activation="relu", 
                 skip_connection=False):
        super().__init__()
        
        # different aggregation functions
        if aggregate == "sum":
            self.aggregate = aggr.SumAggregation()
        elif aggregate == "mean":
            self.aggregate = aggr.MeanAggregation()
        elif aggregate == "max":
            self.aggregate = aggr.MaxAggregation()
        else:
            raise NotImplementedError(f"Aggregation '{aggregate}' not implemented")
        
        self.global_aggregate = aggr.MeanAggregation()
        
        add_edge_fs = 1 if skip_connection else 0
        
        # Graph Net Blocks (see https://arxiv.org/pdf/1806.01261.pdf)
        self.edge_block = MLP([global_features + edge_features + 2 * node_features + add_edge_fs, 
                               hidden_size,
                               edge_features], layer_norm=layer_norm, activation=activation)
        
        # Message passing block if multigraph
        if edge_2_features > 0:
            self.mp_edge_block = MLP([global_features + edge_2_features + 2 * node_features + add_edge_fs, 
                                    hidden_size,
                                    edge_2_features], layer_norm=layer_norm, activation=activation)
        else:
            self.mp_edge_block = None
        
        self.node_block = MLP([global_features + edge_features + edge_2_features + node_features,
                               hidden_size,
                               node_features], layer_norm=layer_norm, activation=activation)
        
        # optional set of blocks for global GNN
        self.global_block = None
        if global_features > 0:
            self.global_block = MLP([edge_features + edge_2_features + node_features + global_features, 
                                     hidden_size,
                                     global_features], layer_norm=layer_norm, activation=activation)
        
    def forward(self, x, edge_index, edge_attr, g=None, edge_index_2=None, edge_attr_2=None):
        row, col = edge_index
        
        if self.global_block is not None:
            assert g is not None, "Need global features for global block"
            
            edge_embedding = self.edge_block(torch.cat([torch.ones(x[row].shape[0], 1, device=device) * g, 
                                                        x[row], x[col], edge_attr], dim=1))
            aggregation = self.aggregate(edge_embedding, row)
            
            if edge_index_2 is not None:
                mp = self.mp_edge_block(torch.cat([torch.ones(x[row].shape[0], 1, device=device) * g, x[row], x[col], edge_attr_2], dim=1))
                agg_features = torch.cat([torch.ones(x.shape[0], 1, device=device) * g, x, aggregation, 
                                          self.aggregate(mp, row)], dim=1)
                mp_global_aggr = torch.cat([g, self.aggregate(mp)], dim=1)
            else:
                agg_features = torch.cat([torch.ones(x.shape[0], 1, device=device) * g, x, aggregation], dim=1)
                mp_global_aggr = g
            
            node_embeddings = self.node_block(agg_features)
            
            # aggregate over all edges and nodes (always mean)
            edge_aggregation_global = self.global_aggregate(edge_embedding)
            node_aggregation_global = self.global_aggregate(node_embeddings)
            
            # compute the new global embedding
            # the old global feature is part of mp_global_aggr
            global_embeddings = self.global_block(torch.cat([node_aggregation_global, 
                                                             edge_aggregation_global,
                                                             mp_global_aggr], dim=1))
            
            return edge_embedding, node_embeddings, global_embeddings
        
        else:
            edge_embedding = self.edge_block(torch.cat([x[row], x[col], edge_attr], dim=1))
            aggregation = self.aggregate(edge_embedding, row)
            
            if edge_index_2 is not None:
                mp = self.mp_edge_block(torch.cat([x[row], x[col], edge_attr_2], dim=1))
                mp_aggregation = self.aggregate(mp, row)
                agg_features = torch.cat([x, aggregation, mp_aggregation], dim=1)
            else:
                agg_features = torch.cat([x, aggregation], dim=1)
            
            node_embeddings = self.node_block(agg_features)
            return edge_embedding, node_embeddings, None


class MLP(nn.Module):
    def __init__(self, width, layer_norm=False, activation="relu", activate_final=False):
        super().__init__()
        width = list(filter(lambda x: x > 0, width))
        assert len(width) >= 2, "Need at least one layer in the network!"

        lls = nn.ModuleList()
        for k in range(len(width)-1):
            lls.append(nn.Linear(width[k], width[k+1], bias=True))
            if k != (len(width)-2) or activate_final:
                if activation == "relu":
                    lls.append(nn.ReLU())
                elif activation == "tanh":
                    lls.append(nn.Tanh())
                elif activation == "leakyrelu":
                    lls.append(nn.LeakyReLU())
                else:
                    raise NotImplementedError(f"Activation '{activation}' not implemented")

        if layer_norm:
            lls.append(nn.LayerNorm(width[-1]))

        self.m = nn.Sequential(*lls)

    def forward(self, x):
        return self.m(x)


class MP_BLOCK(nn.Module):
    # L@L.T matrix multiplication graph layer
    # Aligns the computation of L@L.T - A with the learned updates
    def __init__(self, skip_connections, edge_features, node_features, global_features, hidden_size) -> None:
        super().__init__()
        
        # We use 2 graph nets in order to operate on the upper and lower triangular parts of the matrix
        self.l1 = GraphNet(node_features=node_features, edge_features=edge_features, global_features=global_features, 
                           hidden_size=hidden_size, skip_connection=skip_connections, aggregate="mean")
        self.l2 = GraphNet(node_features=node_features, edge_features=edge_features, global_features=global_features, 
                           hidden_size=hidden_size, aggregate="mean")
    
    def forward(self, l_x, u_x, l_edge_index, u_edge_index, l_edge_attr, u_edge_attr, global_features):

        l_edge_embedding, l_node_embeddings, global_features = self.l1(l_x, l_edge_index, l_edge_attr, g=global_features)
        u_edge_embedding, u_node_embeddings, global_features = self.l2(l_node_embeddings, u_edge_index, u_edge_attr, g=global_features)
        
        return l_edge_embedding, l_node_embeddings, u_edge_embedding, u_node_embeddings, global_features
        

############################
#         NEURALIF         #
############################
class NeuralIF(nn.Module):
    # Neural Incomplete factorization
    def __init__(self, message_passing_steps=3, **kwargs) -> None:
        super().__init__()
        
        self.global_features = kwargs['global_features']
        # node features are augmented with local degree profile
        self.augment_node_features = True
        num_node_features = 8 if self.augment_node_features else 1
        
        self.mps = torch.nn.ModuleList()
        for l in range(message_passing_steps):
            # skip connections are added to all layers except the first one
            self.mps.append(MP_BLOCK(skip_connections=(l!=0),
                                     edge_features=1,
                                     node_features=num_node_features, 
                                     global_features=kwargs['global_features'],
                                     hidden_size=kwargs['latent_size']))
        
    def forward(self, data):
        # ! data could be batched here...(not implemented)
        
        # transform nodes to include more features
        if self.augment_node_features:
            data.x = torch.arange(data.x.size()[0], dtype=ddtype, device=device).unsqueeze(1)
            data = torch_geometric.transforms.LocalDegreeProfile()(data)
            data.x = data.x.to(ddtype)
            
            # diagonal dominance and diagonal decay from the paper
            row, col = data.edge_index
            diag = row == col
            diag_elem = torch.abs(data.edge_attr[diag])
            # remove diagonal elements by setting them to zero
            non_diag_elem = data.edge_attr.clone()
            non_diag_elem[diag] = 0
            
            row_sums = aggr.SumAggregation()(torch.abs(non_diag_elem), row)
            alpha = diag_elem / row_sums
            row_dominance_feature = alpha / (alpha + 1)
            row_dominance_feature = torch.nan_to_num(row_dominance_feature, nan=1.0)
            
            # compute diagonal decay features
            row_max = aggr.MaxAggregation()(torch.abs(non_diag_elem), row)
            alpha = diag_elem / row_max
            row_decay_feature = alpha / (alpha + 1)
            row_decay_feature = torch.nan_to_num(row_decay_feature, nan=1.0)
            
            data.x = torch.cat([data.x, row_dominance_feature, row_decay_feature], dim=1)
        
        # make data lower triangular
        U_data = (ToUpperTriangular()(data))
        # make data upper triangular
        L_data = (ToLowerTriangular()(data))
        
        # get the input data for L
        l_edge_embedding = (L_data.edge_attr).to(ddtype)
        l_node_embedding = L_data.x
        l_index = L_data.edge_index

        # get the input data for U
        u_edge_embedding = (U_data.edge_attr).to(ddtype)
        u_node_embedding = U_data.x
        u_index = U_data.edge_index
        
        # copy the input data (only edges of original matrix A)
        l_a_edges = l_edge_embedding.clone()
        u_a_edges = u_edge_embedding.clone()
        
        if self.global_features > 0:
            global_features = torch.zeros((1, self.global_features), device=device, requires_grad=False)
        else:
            global_features = None
        
        # compute the output of the network
        for i, layer in enumerate(self.mps):
            if i != 0:
                l_edge_embedding = torch.cat([l_edge_embedding, l_a_edges], dim=1)
                #u_edge_embedding = torch.cat([u_edge_embedding, u_a_edges], dim=1)
            
            l_edge_embedding, l_node_embeddings, u_edge_embedding, u_node_embeddings, global_features = layer(l_node_embedding, u_node_embedding, 
                                                                                                              l_index, u_index, 
                                                                                                              l_edge_embedding, u_edge_embedding, global_features)
        
        # transform the output into a matrix
        L = self.transform_output_matrix(l_node_embedding, l_index, l_edge_embedding)
        U = self.transform_output_matrix(u_node_embedding, u_index, u_edge_embedding)

        return L, U, None # no node embedding

    def transform_output_matrix(self, node_x, edge_index, edge_values):
        # force diagonal to be positive
        diag = edge_index[0] == edge_index[1]
        edge_values[diag] = torch.exp(edge_values[diag])
        
        t = torch.sparse_coo_tensor(edge_index, edge_values.squeeze(), size=(node_x.size()[0], node_x.size()[0]))
        
        return t