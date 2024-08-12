import os
import re
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from collections import Counter
from tqdm import tqdm
        

class yelp:
    def __init__(self, args):                                  
        self.args = args
        self.content_filename = ["review_embeddings.npy", "user_embeddings.npy", "business_embeddings.npy"]
        
        self.r_embed = torch.zeros(args.R_n, args.embed_dim, dtype=torch.float32)
        self.u_embed = torch.zeros(args.U_n, args.embed_dim, dtype=torch.float32)
        self.b_embed = torch.zeros(args.B_n, args.embed_dim, dtype=torch.float32)

        self.r_r_edge_index = torch.empty(0)
        self.r_u_edge_index = torch.empty(0)
        self.r_b_edge_index = torch.empty(0) 
        self.u_r_edge_index = torch.empty(0)
        self.u_u_edge_index = torch.empty(0)
        self.u_b_edge_index = torch.empty(0)
        self.b_r_edge_index = torch.empty(0)
        self.b_u_edge_index = torch.empty(0)
        self.b_b_edge_index = torch.empty(0)   
        
        self.b_class = torch.full((args.B_n,), -1, dtype=torch.long)
    
    def read_content_file(self): # r_embed, u_embed_ b_embed

        for f_num in range(len(self.content_filename)):         
            data = np.load(self.args.data_path + self.content_filename[f_num])
		
            if f_num == 0:
                self.r_embed = torch.tensor(data).to(torch.float32) 
            elif f_num == 1:
                self.u_embed = torch.tensor(data).to(torch.float32) 
            else:
                self.b_embed = torch.tensor(data).to(torch.float32)
    
    def read_walk_file(self): # all edges indices based on neighbours in random walks
        
        r_edge_list = []
        u_edge_list = []
        b_edge_list = []
        r_r_edge_index = []
        r_u_edge_index = []
        r_b_edge_index = []
        u_r_edge_index = []
        u_u_edge_index = []
        u_b_edge_index = []
        b_r_edge_index = []
        b_u_edge_index = []
        b_b_edge_index = []
        
        with open(self.args.data_path + "random_walks.txt", 'r') as file:            
            lines = list(tqdm(file, desc='Reading random walks file', leave=True))
                
        for i, line in tqdm(enumerate(lines), desc='Processing lines', leave=True):
            line = line.strip()
            node_type = re.split(':', line)[0][0]
            node_id = int(re.split(':', line)[0][1:])
            neigh_list = re.split(',', re.split(':', line)[1].strip())
            
            r_edge_list = [node for node in neigh_list if node.startswith('r')]
            u_edge_list = [node for node in neigh_list if node.startswith('u')]
            b_edge_list = [node for node in neigh_list if node.startswith('b')]
            
            # Count the frequency of elements starting with 'r', 'u', and 'b'
            r_counts = Counter(r_edge_list)
            u_counts = Counter(u_edge_list)
            b_counts = Counter(b_edge_list)
            # print(a_counts)
            
            r_edge_list = [node for node, count in r_counts.most_common(10)]
            u_edge_list = [node for node, count in u_counts.most_common(10)]
            b_edge_list = [node for node, count in b_counts.most_common(5)]
            # print(a_edge_list)
            
            if node_type == 'r':
                r_r_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in r_edge_list])
                r_u_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in u_edge_list])
                r_b_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in b_edge_list])
            elif node_type == 'u':
                u_r_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in r_edge_list])
                u_u_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in u_edge_list])
                u_b_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in b_edge_list])
            else:
                b_r_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in r_edge_list])
                b_u_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in u_edge_list])
                b_b_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in b_edge_list])
        
        # Concatenate the list of tensors into a single tensor
        self.r_r_edge_index = torch.cat(r_r_edge_index, dim=0).t().contiguous()
        self.r_u_edge_index = torch.cat(r_u_edge_index, dim=0).t().contiguous()
        self.r_b_edge_index = torch.cat(r_b_edge_index, dim=0).t().contiguous()
        self.u_r_edge_index = torch.cat(u_r_edge_index, dim=0).t().contiguous()
        self.u_u_edge_index = torch.cat(u_u_edge_index, dim=0).t().contiguous()
        self.u_b_edge_index = torch.cat(u_b_edge_index, dim=0).t().contiguous()
        self.b_r_edge_index = torch.cat(b_r_edge_index, dim=0).t().contiguous()
        self.b_u_edge_index = torch.cat(b_u_edge_index, dim=0).t().contiguous()
        self.b_b_edge_index = torch.cat(b_b_edge_index, dim=0).t().contiguous()
        
        
    def read_label_file(self):
        
        with open(self.args.data_path + "b_classes.txt", 'r') as file:            
            lines = list(tqdm(file, desc='Reading classes file', leave=True))
        for i, line in enumerate(lines):
                entries =  line.strip().split()
                self.b_class[int(entries[0])] = int(entries[1].strip())
 
       
def input_data(args):
    dataset = yelp(args)
    dataset.read_content_file()
    dataset.read_walk_file()
    dataset.read_label_file()

    data = HeteroData()
    
    data['r'].num_nodes = args.R_n
    data['u'].num_nodes = args.U_n
    data['b'].num_nodes = args.B_n

    data['r_embed'].x = dataset.r_embed
    data['u_embed'].x = dataset.u_embed 
    data['b_embed'].x = dataset.b_embed
    
    data['b'].y = dataset.b_class

    data['r', 'walk', 'r'].edge_index = dataset.r_r_edge_index
    data['r', 'walk', 'u'].edge_index = dataset.r_u_edge_index
    data['r', 'walk', 'b'].edge_index = dataset.r_b_edge_index
    
    data['u', 'walk', 'r'].edge_index = dataset.u_r_edge_index
    data['u', 'walk', 'u'].edge_index = dataset.u_u_edge_index
    data['u', 'walk', 'b'].edge_index = dataset.u_b_edge_index
    
    data['b', 'walk', 'r'].edge_index = dataset.b_r_edge_index
    data['b', 'walk', 'u'].edge_index = dataset.b_u_edge_index
    data['b', 'walk', 'b'].edge_index = dataset.b_b_edge_index
    
    return data
    
 
    



# Function for heterogenous neighbour aggregation
def aggregate(x, edge_index, num_nodes): 
    # Separate source and target nodes from the edge index
    source_nodes, target_nodes = edge_index[0], edge_index[1]

    # Aggregate features for each neighbour using scatter_add
    # num_source = torch.max(source_nodes, dim = 0).values.item()
    aggr_features = torch.zeros(num_nodes, x.size(1))
    aggr_features.index_add_(0, source_nodes, x[target_nodes])

    # Normalize the aggregated features
    row_sum = torch.bincount(source_nodes, minlength=num_nodes).float().clamp(min=1)
    aggr_features /= row_sum.view(-1, 1)

    return aggr_features