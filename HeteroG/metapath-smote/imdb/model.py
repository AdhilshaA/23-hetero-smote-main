import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class Content_Agg(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Content_Agg, self).__init__()
        self.fc = nn.Linear(384, embed_dim)
        self.dropout = dropout

    def forward(self, embed_list):
        
        # Concatenate the embeddings
        x = F.normalize(embed_list[0] + embed_list[1] + embed_list[2], p=2, dim = -1)
        x = F.relu(self.fc(x))
        x = F.dropout(x, self.dropout, training=self.training) 
        return x


# Layer for heterogenous neighbour aggregation
class Neigh_Agg(nn.Module): 
    def __init__(self, embed_dim, dropout):
        super(Neigh_Agg, self).__init__()
        self.aggregation_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index, num_node, edge_weight):
        # Separate source and target nodes from the edge index
        source_nodes, target_nodes = edge_index[0], edge_index[1]

        # Apply ReLU activation to target features through the linear layer
        x_target = F.relu(self.aggregation_layer(x))

        # Multiply target features with edge weights
        x_target_weighted = x_target[target_nodes] * edge_weight.view(-1, 1)

        # Aggregate weighted features for each source using scatter_add
        aggr_features = torch.zeros(num_node, x.size(1), device=x.device)
        aggr_features.index_add_(0, source_nodes, x_target_weighted)

        # Normalize the aggregated features
        row_sum = torch.bincount(source_nodes, weights=edge_weight, minlength=num_node).float().clamp(min=1)
        aggr_features /= row_sum.view(-1, 1)

        # Apply dropout
        aggr_features = F.dropout(aggr_features, self.dropout, training=self.training)

        return aggr_features

class Het_classify(nn.Module):
    def __init__(self, embed_dim, nclass, dropout):
        super(Het_classify, self).__init__()
        self.m_het = Neigh_Agg(embed_dim, dropout)
        self.mlp = nn.Linear(embed_dim, nclass)  
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, edge_index, num_node, edge_weight):
        x = self.m_het(x, edge_index, num_node, edge_weight)
        x = torch.relu(x)
        x = self.mlp(x)

        return x
    

class EdgePredictor(nn.Module):
    def __init__(self, nembed, dropout=0.1):
        super(EdgePredictor, self).__init__()
        self.dropout = dropout
        self.lin = nn.Linear(nembed, nembed)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.lin.weight,std=0.05)

    def forward(self, x):
        x = self.lin(x)
        result = torch.mm(x, x.transpose(-1, -2))
        adj_out = torch.sigmoid(result)         # Apply sigmoid along dim=1 (rows)
        return adj_out