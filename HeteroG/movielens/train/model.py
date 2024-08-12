import torch
import torch.nn as nn
import torch.nn.functional as F

class Content_Agg(nn.Module):
    def __init__(self, embed_dim, attr_size, dropout):
        super(Content_Agg, self).__init__()
        self.fc = nn.Linear(attr_size, embed_dim)
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
        # self.num_node = num_node
    
    def forward(self, x, edge_index, num_node):
        
        # Separate source and target nodes from the edge index
        source_nodes, target_nodes = edge_index[0], edge_index[1]

        # Apply ReLU activation to target features through the linear layer
        x_target = F.relu(self.aggregation_layer(x))

        # Aggregate target features for each source using scatter_add
        aggr_features = torch.zeros(num_node, x.size(1), device=x.device)
        # print(x_target.shape, max(source_nodes), max(target_nodes), aggr_features.shape)
        aggr_features.index_add_(0, source_nodes, x_target[target_nodes])

        # Normalize the aggregated features
        row_sum = torch.bincount(source_nodes, minlength=num_node).float().clamp(min=1)
        aggr_features /= row_sum.view(-1, 1)
        aggr_features = F.dropout( aggr_features, self.dropout, training=self.training)

        return aggr_features
    

class Het_Agg(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Het_Agg, self).__init__()
        self.u_aggregate = Neigh_Agg(embed_dim, dropout)
        self.m_aggregate = Neigh_Agg(embed_dim, dropout)

        self.u = nn.Parameter(torch.randn(2 * embed_dim, 1))
        nn.init.normal_(self.u, mean=0, std=1)  # Normalize self.u
        self.linear = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, x_list, edges_list, x_node, num_node):
        u_aggr = self.u_aggregate(x_list[0], edges_list[0], num_node)
        m_aggr = self.m_aggregate(x_list[1], edges_list[1], num_node)
        
        # Concatenate x_node with each aggregation
        u_aggr_cat = torch.cat((u_aggr, x_node), dim=-1)
        m_aggr_cat = torch.cat((m_aggr, x_node), dim=-1)
        
        # Matrix multiplication with learnable parameter u
        u_scores = torch.exp(F.leaky_relu(torch.matmul(u_aggr_cat, self.u)))
        m_scores = torch.exp(F.leaky_relu(torch.matmul(m_aggr_cat, self.u)))
        
        # Sum of scores for all types
        sum_scores = u_scores + m_scores
        
        # Calculate attention weights using softmax
        u_weights = u_scores / sum_scores
        m_weights = m_scores / sum_scores
        
        # Combine embeddings with attention weights
        combined_aggr = ((u_weights * u_aggr) +  (m_weights * m_aggr))
        
        # Concatenate the combined aggregation with x_node
        combined_aggr = torch.cat((x_node, combined_aggr), dim=-1)
        
        # Apply learnable linear layer to get the final aggregated embedding
        final_aggr = F.normalize(F.relu(self.linear(combined_aggr)), p=2, dim=-1)

        return final_aggr
 
    
class Het_ConEn(nn.Module):
    def __init__(self, embed_dim, args, dropout):
        super(Het_ConEn, self).__init__()
        
        self.u_con = Content_Agg(embed_dim, args.U_emsize, dropout)
        self.m_con = Content_Agg(embed_dim, args.M_emsize, dropout)
        self.u_cont = torch.empty(0)
        self.m_cont = torch.empty(0)
        
    def forward(self, data):
        
        u_embed_list = [data['u_embed'].x, data['u_net_embed'].x]
        m_embed_list = [data['m_embed'].x, data['m_net_embed'].x]
        self.u_cont = self.u_con(u_embed_list)
        self.m_cont = self.m_con(m_embed_list)
        
        return [self.u_cont, self.m_cont]
  

class Het_NetEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Het_NetEn, self).__init__()
        
        self.u_het = Het_Agg(embed_dim, dropout)
        self.m_het = Het_Agg(embed_dim, dropout)
        
    def forward(self, x_list, data):

        u_edges_list = [data['u', 'walk', 'u'].edge_index, data['u', 'walk', 'm'].edge_index]
        m_edges_list = [data['m', 'walk', 'u'].edge_index, data['m', 'walk', 'm'].edge_index]
        
        x_list[0] = self.u_het(x_list, u_edges_list, x_list[0], x_list[0].size(0)) 
        x_list[1] = self.m_het(x_list, m_edges_list, x_list[1], x_list[1].size(0))
       
        return x_list

    

class Edge_classify(nn.Module):
    def __init__(self, embed_dim, nclass, dropout):
        super(Edge_classify, self).__init__()
        self.u_het = Het_Agg(embed_dim, dropout)
        self.m_het = Het_Agg(embed_dim, dropout)
        self.lin = nn.Linear(embed_dim, 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.mlp = nn.Sequential(nn.ReLU(),nn.Linear(64, nclass))
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.lin.weight,std=0.05)

    def forward(self, x_list, edges_list, target_edge, x, os_mode):
        
        if os_mode != 'edge_sm':
            x_list[0] = self.u_het(x_list, edges_list[0], x_list[0], x_list[0].size(0)) 
            x_list[1] = self.m_het(x_list, edges_list[1], x_list[1], x_list[1].size(0))
            x = x_list[0][target_edge[:, 0]] * x_list[1][target_edge[:, 1]] 
        
        x = self.lin(x)    
        x = self.batch_norm(x)
        x = self.mlp(x)
        x = F.dropout(x, self.dropout, training=self.training)

        return x
    
    
    


# class Het_classify(nn.Module):
#     def __init__(self, embed_dim, nclass, dropout):
#         super(Het_classify, self).__init__()
#         self.u_het = Het_Agg(embed_dim, dropout)
#         self.mlp = nn.Linear(embed_dim, nclass)    
#         self.dropout = dropout
#         self.reset_parameters()
        
#     def reset_parameters(self):
#         nn.init.normal_(self.mlp.weight,std=0.05)

#     def forward(self, x_list, edge_list):
#         x = self.u_het(x_list, edge_list, x_list[0], x_list[0].size(0))
#         x = torch.relu(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.mlp(x)

#         return x
    

# class encoder2(nn.Module):
#     def __init__(self, embed_dim, dropout):
#         super(encoder2, self).__init__()
#         self.m_het = Het_Agg(embed_dim, dropout)

#     def forward(self, x_list, edge_list):
#         x = self.m_het(x_list, edge_list, x_list[1], x_list[1].size(0))
#         return [x_list[0], x, x_list[2]]
    
    

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
    
    