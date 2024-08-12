import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, MessagePassing

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden)  # GCN layer 1
        self.conv2 = GCNConv(hidden, num_classes)   # GCN layer 2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
#gcn_encode
class GCN_En(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout):
        super(GCN_En, self).__init__()

        self.gc1 = GCNConv(num_features, hidden_dim)  # GCN layer 1
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        return x
    
    

class GSAGEModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GSAGEModel, self).__init__()
        
        self.sage1 = SAGEConv(num_features, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, num_classes)
        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class Sage_En(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout):
        super(Sage_En, self).__init__()
        
        self.sage1 = SAGEConv(num_features, hidden_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Sage_Classifier(nn.Module):
    def __init__(self, nembed, hdim, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SAGEConv(nembed, hdim)
        self.mlp = nn.Linear(hdim, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x
    

# class EdgePredictor(MessagePassing):
#     def __init__(self, num_features):
#         super(EdgePredictor, self).__init__(aggr='add')
#         self.lin = nn.Linear(num_features, num_features)  # Linear layer for edge prediction

#     def forward(self, x, edge_index):
#         edge_scores = self.lin(x)
#         x = self.propagate(edge_index, x=x)
#         edge_probs = F.sigmoid(edge_scores)

#         return edge_probs
    
    
class EdgePredictor(nn.Module):
    def __init__(self, nembed, dropout=0.1):
        super(EdgePredictor, self).__init__()
        self.dropout = dropout
        self.lin = nn.Linear(nembed, nembed)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.lin.weight,std=0.05)

    def forward(self, node_embed):
        
        combine = self.lin(node_embed)
        print(combine.shape)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1,-2)))

        return adj_out




