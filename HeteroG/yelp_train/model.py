import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class Content_Agg(nn.Module):
    def __init__(self, embed_dim, num_embed, dropout):
        super(Content_Agg, self).__init__()
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
        self.r_aggregate = Neigh_Agg(embed_dim, dropout)
        self.u_aggregate = Neigh_Agg(embed_dim, dropout)
        self.b_aggregate = Neigh_Agg(embed_dim, dropout)

        self.u = nn.Parameter(torch.randn(2 * embed_dim, 1))
        nn.init.normal_(self.u, mean=0, std=1)  # Normalize self.u
        self.linear = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, x_list, edges_list, x_node, num_node):
        r_aggr = self.r_aggregate(x_list[0], edges_list[0], num_node)
        u_aggr = self.u_aggregate(x_list[1], edges_list[1], num_node)
        b_aggr = self.b_aggregate(x_list[2], edges_list[2], num_node)
        
        # Concatenate x_node with each aggregation
        r_aggr_cat = torch.cat((r_aggr, x_node), dim=-1)
        u_aggr_cat = torch.cat((u_aggr, x_node), dim=-1)
        b_aggr_cat = torch.cat((b_aggr, x_node), dim=-1)
        
        # Matrix multiplication with learnable parameter u
        r_scores = torch.exp(F.leaky_relu(torch.matmul(r_aggr_cat, self.u)))
        u_scores = torch.exp(F.leaky_relu(torch.matmul(u_aggr_cat, self.u)))
        b_scores = torch.exp(F.leaky_relu(torch.matmul(b_aggr_cat, self.u)))
        
        # Sum of scores for all types
        sum_scores = r_scores + u_scores + b_scores
        
        # Calculate attention weights using softmax
        r_weights = r_scores / sum_scores
        u_weights = u_scores / sum_scores
        b_weights = b_scores / sum_scores
        
        # Combine embeddings with attention weights
        combined_aggr = ((r_weights * r_aggr) +  (u_weights * u_aggr) + (b_weights * b_aggr))
        
        # Concatenate the combined aggregation with x_node
        combined_aggr = torch.cat((x_node, combined_aggr), dim=-1)
        
        # Apply learnable linear layer to get the final aggregated embedding
        final_aggr = F.normalize(F.relu(self.linear(combined_aggr)), p=2, dim=-1)
        
        return final_aggr
 
    
class Het_ConEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Het_ConEn, self).__init__()
        
        self.r_con = Content_Agg(embed_dim, 1, dropout)
        self.u_con = Content_Agg(embed_dim, 1, dropout)
        self.b_con = Content_Agg(embed_dim, 1, dropout)
        
        self.r_cont = torch.empty(0)
        self.u_cont = torch.empty(0)
        self.b_cont = torch.empty(0)
        
    def forward(self, data):
        
        r_embed_list = [data['r_embed'].x]
        u_embed_list = [data['u_embed'].x]
        b_embed_list = [data['b_embed'].x]
        
        self.r_cont = self.r_con(r_embed_list)
        self.u_cont = self.u_con(u_embed_list)
        self.b_cont = self.b_con(b_embed_list)
        
        return [self.b_cont, self.u_cont, self.r_cont]
            

class Het_NetEn(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(Het_NetEn, self).__init__()
        
        self.b_het = Het_Agg(embed_dim, dropout)
        self.u_het = Het_Agg(embed_dim, dropout)
        self.r_het = Het_Agg(embed_dim, dropout)
        
        # self.r_cont = torch.empty(0)
        # self.u_cont = torch.empty(0)
        # self.b_cont = torch.empty(0)
        
    def forward(self, x_list, data):
        
        # self.r_cont = x_list[0]
        # self.u_cont = x_list[1]
        # self.b_cont = x_list[2]

        r_edges_list = [data['r', 'walk', 'b'].edge_index, data['r', 'walk', 'u'].edge_index, data['r', 'walk', 'r'].edge_index]
        u_edges_list = [data['u', 'walk', 'b'].edge_index, data['u', 'walk', 'u'].edge_index, data['u', 'walk', 'r'].edge_index]
        b_edges_list = [data['b', 'walk', 'b'].edge_index, data['b', 'walk', 'u'].edge_index, data['b', 'walk', 'r'].edge_index]

        x_list[0] = self.b_het(x_list, b_edges_list, x_list[0], x_list[0].size(0)) 
        x_list[1] = self.u_het(x_list, u_edges_list, x_list[1], x_list[1].size(0))
        x_list[2] = self.r_het(x_list, r_edges_list, x_list[2], x_list[2].size(0))
        
        return x_list
        
        # x_list = [self.r_cont, self.u_cont, self.d_cont]

        # self.r_cont = self.r_het(x_list, r_edges_list, self.r_cont, self.r_cont.size(0)) 
        # self.u_cont = self.a_het(x_list, u_edges_list, self.u_cont, self.u_cont.size(0))
        # self.b_cont = self.b_het(x_list, b_edges_list, self.b_cont, self.b_cont.size(0))
        
        # return [self.r_cont, self.u_cont, self.b_cont]
     

    
class Het_classify(nn.Module):
    def __init__(self, embed_dim, nclass, dropout):
        super(Het_classify, self).__init__()
        self.b_het = Het_Agg(embed_dim, dropout)
        self.mlp = nn.Linear(embed_dim, nclass)    
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x_list, edge_list):
        x = self.b_het(x_list, edge_list, x_list[0], x_list[0].size(0))
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x
    
    
    
    
class Classifier(nn.Module):
    def __init__(self, nembed, nclass, dropout):
        super(Classifier, self).__init__()
        self.sage1 = SAGEConv(nembed, nembed) 
        self.mlp = nn.Linear(nembed, nclass)    
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