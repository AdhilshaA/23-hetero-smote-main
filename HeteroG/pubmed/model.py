import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class Content_Agg(nn.Module):
    def __init__(self, embed_dim, num_embed, dropout):
        super(Content_Agg, self).__init__()
        self.fc = nn.Linear(embed_dim * num_embed, embed_dim)
        self.dropout = dropout

    def forward(self, embed_list):
        # Concatenate the embeddings
        x = torch.cat(embed_list, dim=-1)
        x = F.normalize(F.relu(self.fc(x)), p=2, dim = -1)
        x = F.dropout(x, self.dropout, training=self.training) 
        return x


# Layer for heterogenous neighbour aggregation
class Neigh_Agg(nn.Module): 
    def __init__(self, embed_dim, dropout):
        super(Neigh_Agg, self).__init__()
        self.aggregation_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index, num_node):
        
        # Separate source and target nodes from the edge index
        source_nodes, target_nodes = edge_index[0], edge_index[1]

        # Apply ReLU activation to target features through the linear layer
        x_target = F.relu(self.aggregation_layer(x))

        # Aggregate target features for each source using scatter_add
        aggr_features = torch.zeros(num_node, x.size(1), device=x.device)
        aggr_features.index_add_(0, source_nodes, x_target[target_nodes])

        # Normalize the aggregated features
        row_sum = torch.bincount(source_nodes, minlength=num_node).float().clamp(min=1)
        aggr_features /= row_sum.view(-1, 1)
        aggr_features = F.dropout( aggr_features, self.dropout, training=self.training)

        return aggr_features
    

class Het_Agg(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Het_Agg, self).__init__()
        self.g_aggregate = Neigh_Agg(embed_dim, dropout)
        self.d_aggregate = Neigh_Agg(embed_dim, dropout)
        self.c_aggregate = Neigh_Agg(embed_dim, dropout)
        self.s_aggregate = Neigh_Agg(embed_dim, dropout)

        self.u = nn.Parameter(torch.randn(2 * embed_dim, 1))
        nn.init.normal_(self.u, mean=0, std=1)  # Normalize self.u
        self.linear = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, x_list, edges_list, x_node, num_node):
        g_aggr = self.g_aggregate(x_list[0], edges_list[0], num_node)
        d_aggr = self.d_aggregate(x_list[1], edges_list[1], num_node)
        c_aggr = self.c_aggregate(x_list[2], edges_list[2], num_node)
        s_aggr = self.s_aggregate(x_list[3], edges_list[3], num_node)
        
        # Concatenate x_node with each aggregation
        g_aggr_cat = torch.cat((g_aggr, x_node), dim=-1)
        d_aggr_cat = torch.cat((d_aggr, x_node), dim=-1)
        c_aggr_cat = torch.cat((c_aggr, x_node), dim=-1)
        s_aggr_cat = torch.cat((s_aggr, x_node), dim=-1)
        
        # Matrix multiplication with learnable parameter u
        g_scores = torch.exp(F.leaky_relu(torch.matmul(g_aggr_cat, self.u)))
        d_scores = torch.exp(F.leaky_relu(torch.matmul(d_aggr_cat, self.u)))
        c_scores = torch.exp(F.leaky_relu(torch.matmul(c_aggr_cat, self.u)))
        s_scores = torch.exp(F.leaky_relu(torch.matmul(s_aggr_cat, self.u)))
        
        # Sum of scores for all types
        sum_scores = g_scores + d_scores + c_scores + s_scores
        
        # Calculate attention weights using softmax
        g_weights = g_scores / sum_scores
        d_weights = d_scores / sum_scores
        c_weights = c_scores / sum_scores
        s_weights = s_scores / sum_scores
        
        # Combine embeddings with attention weights
        combined_aggr = ((g_weights * g_aggr) +  (d_weights * d_aggr) + (c_weights * c_aggr) + (s_weights * s_aggr))
        
        # Concatenate the combined aggregation with x_node
        combined_aggr = torch.cat((x_node, combined_aggr), dim=-1)
        
        # Apply learnable linear layer to get the final aggregated embedding
        final_aggr = F.normalize(F.relu(self.linear(combined_aggr)), p=2, dim=-1)
        
        return final_aggr
 
 
class Het_ConEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Het_ConEn, self).__init__()

        self.g_con = Content_Agg(embed_dim, 1, dropout)
        self.d_con = Content_Agg(embed_dim, 1, dropout)
        self.c_con = Content_Agg(embed_dim, 1, dropout)
        self.s_con = Content_Agg(embed_dim, 1, dropout)
        
        self.g_cont = torch.empty(0)
        self.d_cont = torch.empty(0)
        self.c_cont = torch.empty(0)
        self.s_cont = torch.empty(0)
        
    def forward(self, data):
        
        g_embed_list = [data['g'].x]
        d_embed_list = [data['d'].x]
        c_embed_list = [data['c'].x]
        s_embed_list = [data['s'].x]
        
        self.g_cont = self.g_con(g_embed_list)
        self.d_cont = self.d_con(d_embed_list)
        self.c_cont = self.c_con(c_embed_list)
        self.s_cont = self.s_con(s_embed_list)
        
        return [self.g_cont, self.d_cont, self.c_cont, self.s_cont]
            

class Het_NetEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Het_NetEn, self).__init__()
        
        self.g_het = Het_Agg(embed_dim, dropout)
        self.d_het = Het_Agg(embed_dim, dropout)
        self.c_het = Het_Agg(embed_dim, dropout)
        self.s_het = Het_Agg(embed_dim, dropout)
        
    def forward(self, x_list, data):

        g_edges_list = [data['g', 'walk', 'g'].edge_index, data['g', 'walk', 'd'].edge_index, data['g', 'walk', 'c'].edge_index, data['g', 'walk', 's'].edge_index]
        d_edges_list = [data['d', 'walk', 'g'].edge_index, data['d', 'walk', 'd'].edge_index, data['d', 'walk', 'c'].edge_index, data['d', 'walk', 's'].edge_index]
        c_edges_list = [data['c', 'walk', 'g'].edge_index, data['c', 'walk', 'd'].edge_index, data['c', 'walk', 'c'].edge_index, data['c', 'walk', 's'].edge_index]
        s_edges_list = [data['s', 'walk', 'g'].edge_index, data['s', 'walk', 'd'].edge_index, data['s', 'walk', 'c'].edge_index, data['s', 'walk', 's'].edge_index]
        
        x_list[1] = self.d_het(x_list, d_edges_list, x_list[1], x_list[1].size(0))
        x_list[0] = self.g_het(x_list, g_edges_list, x_list[0], x_list[0].size(0)) 
        x_list[2] = self.c_het(x_list, c_edges_list, x_list[2], x_list[2].size(0))
        x_list[3] = self.s_het(x_list, s_edges_list, x_list[3], x_list[3].size(0))
        
        return x_list
     

    
class Het_classify(nn.Module):
    def __init__(self, embed_dim, nclass, dropout):
        super(Het_classify, self).__init__()
        self.d_het = Het_Agg(embed_dim, dropout)
        self.mlp = nn.Linear(embed_dim, nclass)    
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x_list, edge_list):
     
        x = self.d_het(x_list, edge_list, x_list[1], x_list[1].size(0))
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x
    
    

class EdgePredictor(nn.Module):
    def __init__(self, nembed, dropout=0.1):
        super(EdgePredictor, self).__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(nembed, nembed)
        self.lin2 = nn.Linear(nembed, nembed)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.lin1.weight,std=0.05)
        nn.init.normal_(self.lin2.weight,std=0.05)

    def forward(self, node_embed1, node_embed2):
        
        combine1 = self.lin1(node_embed1)
        combine2 = self.lin2(node_embed2)
        result = torch.mm(combine1, combine2.transpose(-1, -2))

        adj_out = torch.sigmoid(result)         # Apply sigmoid along dim=1 (rows)
        return adj_out