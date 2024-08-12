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

    
class GCN(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(GCN, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = dropout

    def forward(self, x, adj, adj_w):
        adj += adj_w
        support = torch.mm(x, self.weight)
        output = F.normalize(torch.spmm(adj, support), p=2, dim = -1)
        output = F.dropout(output, self.dropout, training=self.training)
        return output

class Het_classify(nn.Module):
    def __init__(self, embed_dim, nclass, dropout):
        super(Het_classify, self).__init__()
        self.m_het = GCN(embed_dim, dropout)
        self.mlp = nn.Linear(embed_dim, nclass)  
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj, adj_w):
        x = self.m_het(x, adj, adj_w)
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
    
