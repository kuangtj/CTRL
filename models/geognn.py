import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, MessagePassing
import numpy as np

NUM_ATOMIC_NUM             = 120 + 2   # （0~118）
NUM_CHIRALITY              = 4 + 2     # unlabel / R / S
NUM_IS_AROMATIC            = 3 + 2     # False / True
NUM_HYBRIDIZATION          = 8 + 2     # e.g. SP, SP2, SP3, SP3D, SP2D, OTHER
NUM_DEGREE                 = 12 + 2    # 0~10
NUM_NUM_HS                 = 6 + 2     # 0~4
NUM_FORMAL_CHARGE          = 12 + 2     # -2, -1, 0, 1, 2
NUM_IS_IN_RING             = 3 + 2     # False / True
NUM_NUM_RADICAL_ELECTRONS  = 6 + 2     # 0~2

# —— 键属性类别数 —— #
NUM_BOND_TYPE      = 5 + 3     # single, double, triple, aromatic, self-loop
NUM_BOND_DIRECTION = 4 + 3     # none, end-to-end, any-one-to-one
NUM_BOND_IN_RING   = 3 + 3     # False / True
NUM_BOND_CONJ      = 3 + 3     # False / True
NUM_BOND_STEREO    = 5 + 3     # none, any, E, Z

class GraphNorm(nn.Module):
    """
    Graph normalization: each node feature is divided by sqrt(num_nodes) per graph.

    Args:
        None

    Forward Args:
        x (Tensor): Node features, shape [num_nodes, feature_size]
        batch (LongTensor): Batch vector mapping each node to its graph ID, shape [num_nodes]

    Returns:
        Tensor: Normalized node features, same shape as x
    """
    def __init__(self):
        super(GraphNorm, self).__init__()

    def forward(self, x, batch):
        # count nodes per graph
        ones = torch.ones((x.size(0), 1), device=x.device)
        counts = global_add_pool(ones, batch)    # [num_graphs, 1]
        norm = torch.sqrt(counts)                # [num_graphs, 1]
        # expand to nodes
        norm_nodes = norm[batch]                 # [num_nodes, 1]
        return x / norm_nodes


class MeanPool(nn.Module):
    """
    Mean pooling over nodes per graph.

    Args:
        None

    Forward Args:
        x (Tensor): Node features [num_nodes, feature_size]
        batch (LongTensor): Batch vector [num_nodes]

    Returns:
        Tensor: Pooled graph features [num_graphs, feature_size]
    """
    def __init__(self):
        super(MeanPool, self).__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN) layer with edge features.

    Each message is: message = x_j + e_ij
    Aggregation: sum
    Update: MLP
    """
    def __init__(self, hidden_size):
        super(GINConv, self).__init__(aggr='add')  # sum aggregation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x (Tensor): Node feature matrix [num_nodes, hidden_size]
            edge_index (LongTensor): Graph connectivity [2, num_edges]
            edge_attr (Tensor): Edge features [num_edges, hidden_size]
        """
        # propagate will call message() and aggregate
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.mlp(out)

    def message(self, x_j, edge_attr):
        # x_j is source node features for each edge
        return x_j + edge_attr

    def update(self, aggr_out):
        # aggr_out is the aggregated messages per node
        return aggr_out

class GeoGNNBlock(nn.Module):
    """
    GeoGNN Block in PyTorch / PyG
    """
    def __init__(self, embed_dim, dropout_rate, last_act):
        super().__init__()
        self.gnn         = GINConv(embed_dim)
        self.norm        = nn.LayerNorm(embed_dim)
        self.graph_norm  = GraphNorm()
        self.act         = nn.ReLU() if last_act else nn.Identity()
        self.dropout     = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr, batch):
        # x:       [N, D]
        # edge_index: [2, E]
        # edge_attr:  [E, D]
        # batch:      [N]  
        out = self.gnn(x, edge_index, edge_attr)
        out = self.norm(out)
        out = self.graph_norm(out, batch)   
        out = self.act(out)
        out = self.dropout(out)
        return out + x

class atom_embedding(nn.Module):
    """
    GeoGNN Block in PyTorch / PyG
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.atom_embedding_list      = nn.ModuleList()
        self.atom_embedding_list.append(nn.Embedding(NUM_ATOMIC_NUM,            emb_dim))
        self.atom_embedding_list.append(nn.Embedding(NUM_CHIRALITY,             emb_dim))
        self.atom_embedding_list.append(nn.Embedding(NUM_IS_AROMATIC,           emb_dim))
        self.atom_embedding_list.append(nn.Embedding(NUM_HYBRIDIZATION,         emb_dim))
        self.atom_embedding_list.append(nn.Embedding(NUM_DEGREE,                emb_dim))
        self.atom_embedding_list.append(nn.Embedding(NUM_NUM_HS,                emb_dim))
        self.atom_embedding_list.append(nn.Embedding(NUM_FORMAL_CHARGE,         emb_dim))
        self.atom_embedding_list.append(nn.Embedding(NUM_IS_IN_RING,            emb_dim))
        self.atom_embedding_list.append(nn.Embedding(NUM_NUM_RADICAL_ELECTRONS, emb_dim))

    def forward(self, x):
    
        h = 0
        for i, name in enumerate(self.atom_embedding_list):
            h = h + self.atom_embedding_list[i](x[:,i])
            
        return h

class bond_embedding(nn.Module):
    """
    GeoGNN Block in PyTorch / PyG
    """
    def __init__(self, emb_dim):
        super().__init__()

        self.bond_embedding_list      = nn.ModuleList()
        self.bond_embedding_list.append(nn.Embedding(NUM_BOND_TYPE,      emb_dim))
        self.bond_embedding_list.append(nn.Embedding(NUM_BOND_DIRECTION, emb_dim))
        self.bond_embedding_list.append(nn.Embedding(NUM_BOND_IN_RING,   emb_dim))
        self.bond_embedding_list.append(nn.Embedding(NUM_BOND_CONJ,      emb_dim))
        self.bond_embedding_list.append(nn.Embedding(NUM_BOND_STEREO,    emb_dim))

    def forward(self, edge_attr):

        e = 0
        for i, name in enumerate(self.bond_embedding_list):
            e = e + self.bond_embedding_list[i](edge_attr[:,i])
            
        return e

class RBF(nn.Module):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, embed_dim, dtype=torch.float32, device=None):
        super(RBF, self).__init__()
        self.centers = torch.tensor(centers, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: [1, n_centers]
#        self.centers = centers  # Shape: [1, n_centers]
        self.gamma = gamma
        self.linear = nn.Linear(len(centers), embed_dim)
    
    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = x.view(-1, 1)  # Reshaping x to (-1, 1)
        x_1 = x - self.centers
#        x_1 = x - self.centers.to(x.device)
        x_2 = x_1**2
        x_3 = torch.exp(-self.gamma * x_2)
        x_4 = self.linear(x_3)
        return x_4  # RBF kernel
        
class GeoGNNModel(nn.Module):
    """
    GeoGNN Model in PyTorch / PyG
    """
    def __init__(self, embed_dim, layer_num, device=None):
        super().__init__()
        self.embed_dim     = embed_dim
        self.dropout_rate  = 0.2
        self.layer_num     = layer_num

        self.init_atom_embedding = atom_embedding(self.embed_dim)
        self.init_bond_embedding = bond_embedding(self.embed_dim)
        self.init_bond_float_rbf = RBF(np.arange(0, 2, 0.1), 10.0, self.embed_dim, device=device)

        # lists of per-layer modules
        self.bond_embedding_list      = nn.ModuleList()
        self.bond_float_rbf_list      = nn.ModuleList()
        self.bond_angle_rbf_list      = nn.ModuleList()
        self.atom_bond_block_list     = nn.ModuleList()
        self.bond_angle_block_list    = nn.ModuleList()

        for i in range(self.layer_num):
            self.bond_embedding_list.append(bond_embedding(self.embed_dim))
            self.bond_float_rbf_list.append(RBF(np.arange(0, 2, 0.1), 10.0, self.embed_dim, device=device))
            self.bond_angle_rbf_list.append(RBF(np.arange(0, np.pi, 0.1), 10.0, self.embed_dim, device=device))
            last_act = (i != self.layer_num - 1)
            self.atom_bond_block_list.append(
                GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act))
            self.bond_angle_block_list.append(
                GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act))

        # readout
        self.graph_pool = MeanPool()

    @property
    def node_dim(self):
        return self.embed_dim

    @property
    def graph_dim(self):
        return self.embed_dim

    def forward(self, data_2d, data_3d):
        # atom_data: torch_geometric.data.Data 
        #   .x        [N,]
        #   .edge_attr[E,]
        #   .edge_index[2, E]
        #   .batch    [N,]
        x = self.init_atom_embedding(data_2d.x)
        b_emb = self.init_bond_embedding(data_2d.edge_attr)
        e = b_emb + self.init_bond_float_rbf(data_2d.bond_lengths)

        x_list = [x]
        e_list = [e]

        for i in range(self.layer_num):
            # atom–bond 
            x = self.atom_bond_block_list[i](
                x_list[-1], data_2d.edge_index, e_list[-1], data_2d.batch)

            eb = self.bond_embedding_list[i](data_2d.edge_attr)
            eb = eb + self.bond_float_rbf_list[i](data_2d.bond_lengths)
            ea = self.bond_angle_rbf_list[i](data_3d.edge_attr)
            e = self.bond_angle_block_list[i](eb, data_3d.edge_index, ea, batch=data_3d.batch)
            x_list.append(x)
            e_list.append(e)

        node_repr  = x_list[-1]
        edge_repr  = e_list[-1]
        graph_repr = self.graph_pool(node_repr, data_2d.batch)
        return node_repr, graph_repr