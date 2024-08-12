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
    def __init__(self, embed_dim, num_node, dropout):
        super(Neigh_Agg, self).__init__()
        self.aggregation_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.num_node = num_node
    
    def forward(self, x, edge_index):
        
        # Separate source and target nodes from the edge index
        source_nodes, target_nodes = edge_index[0], edge_index[1]

        # Apply ReLU activation to target features through the linear layer
        x_target = F.relu(self.aggregation_layer(x))

        # Aggregate target features for each source using scatter_add
        # num_source = torch.max(source_nodes, dim = 0)
        aggr_features = torch.zeros(self.num_node, x.size(1), device=x.device)
        aggr_features.index_add_(0, source_nodes, x_target[target_nodes])

        # Normalize the aggregated features
        row_sum = torch.bincount(source_nodes, minlength=self.num_node).float().clamp(min=1)
        aggr_features /= row_sum.view(-1, 1)
        aggr_features = F.dropout( aggr_features, self.dropout, training=self.training)

        return aggr_features
    

class Het_Agg(nn.Module):
    def __init__(self, embed_dim, num_node, dropout):
        super(Het_Agg, self).__init__()
        self.a_aggregate = Neigh_Agg(embed_dim, num_node, dropout)
        self.p_aggregate = Neigh_Agg(embed_dim, num_node, dropout)
        self.v_aggregate = Neigh_Agg(embed_dim, num_node, dropout)

        self.u = nn.Parameter(torch.randn(2 * embed_dim, 1))
        nn.init.normal_(self.u, mean=0, std=1)  # Normalize self.u
        self.linear = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, x_list, edges_list, x_node):
        a_aggr = self.a_aggregate(x_list[0], edges_list[0])
        p_aggr = self.p_aggregate(x_list[1], edges_list[1])
        v_aggr = self.v_aggregate(x_list[2], edges_list[2])
        
        # Concatenate x_node with each aggregation
        a_aggr_cat = torch.cat((a_aggr, x_node), dim=-1)
        p_aggr_cat = torch.cat((p_aggr, x_node), dim=-1)
        v_aggr_cat = torch.cat((v_aggr, x_node), dim=-1)
        
        # Matrix multiplication with learnable parameter u
        a_scores = torch.exp(F.leaky_relu(torch.matmul(a_aggr_cat, self.u)))
        p_scores = torch.exp(F.leaky_relu(torch.matmul(p_aggr_cat, self.u)))
        v_scores = torch.exp(F.leaky_relu(torch.matmul(v_aggr_cat, self.u)))
        
        # Sum of scores for all types
        sum_scores = a_scores + p_scores + v_scores
        
        # Calculate attention weights using softmax
        a_weights = a_scores / sum_scores
        p_weights = p_scores / sum_scores
        v_weights = v_scores / sum_scores
        
        # Combine embeddings with attention weights
        combined_aggr = ((a_weights * a_aggr) +  (p_weights * p_aggr) + (v_weights * v_aggr))
        
        # Concatenate the combined aggregation with x_node
        combined_aggr = torch.cat((x_node, combined_aggr), dim=-1)
        
        # Apply learnable linear layer to get the final aggregated embedding
        final_aggr = F.normalize(F.relu(self.linear(combined_aggr)), p=2, dim=-1)
        
        return final_aggr
 
    
class Het_En(nn.Module):
    def __init__(self, node_dim, embed_dim, dropout):
        super(Het_En, self).__init__()
        
        self.a_con = Content_Agg(embed_dim, 2, dropout)
        self.p_con = Content_Agg(embed_dim, 6, dropout)
        self.v_con = Content_Agg(embed_dim, 2, dropout)
        self.a_het = Het_Agg(embed_dim, node_dim[0], dropout)
        self.p_het = Het_Agg(embed_dim, node_dim[1], dropout)
        self.v_het = Het_Agg(embed_dim, node_dim[2], dropout)
        
        self.a_cont = torch.empty(0)
        self.p_cont = torch.empty(0)
        self.v_cont = torch.empty(0)
        
    def forward(self, data):
        
        a_embed_list = [data['a_net_embed'].x, data['a_text_embed'].x]
        p_embed_list = [data['p_title_embed'].x, data['p_abstract_embed'].x, data['p_net_embed'].x, data['p_a_net_embed'].x,
                       data['p_p_net_embed'].x, data['p_v_net_embed'].x]
        v_embed_list = [data['v_net_embed'].x, data['v_text_embed'].x]
        
        self.a_cont = self.a_con(a_embed_list)
        self.p_cont = self.p_con(p_embed_list)
        self.v_cont = self.v_con(v_embed_list)

        a_edges_list = [data['a', 'walk', 'a'].edge_index, data['a', 'walk', 'p'].edge_index, data['a', 'walk', 'v'].edge_index]
        p_edges_list = [data['p', 'walk', 'a'].edge_index, data['p', 'walk', 'p'].edge_index, data['p', 'walk', 'v'].edge_index]
        v_edges_list = [data['v', 'walk', 'a'].edge_index, data['v', 'walk', 'p'].edge_index, data['v', 'walk', 'v'].edge_index]
        
        x_list = [self.a_cont, self.p_cont, self.v_cont]
    
        self.v_cont = self.v_het(x_list, v_edges_list, self.v_cont)
        self.p_cont = self.p_het(x_list, p_edges_list, self.p_cont)
        self.a_cont = self.a_het(x_list, a_edges_list, self.a_cont)
        
        return self.a_cont

    
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