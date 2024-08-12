import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from collections import Counter


class Aminer:
    def __init__(self, args):                                  
        self.args = args
        self.cite_filename = ["p_a_list.txt", "p_p_list.txt", "p_v_list.txt"]
        self.content_filename = "node.dat"
        self.neigh_filename = ["a_p_list.txt", "v_p_list_train.txt"]
        
        self.g_embed = torch.zeros(args.G_n, args.embed_d, dtype=torch.float32)
        self.d_embed = torch.zeros(args.D_n, args.embed_d, dtype=torch.float32)
        self.c_embed = torch.zeros(args.C_n, args.embed_d, dtype=torch.float32)
        self.d_embed = torch.zeros(args.D_n, args.embed_d, dtype=torch.float32)
        
        self.p_abstract_embed = torch.zeros(args.P_n, args.embed_d, dtype=torch.float32)
        
        self.p_net_embed = torch.zeros(args.P_n, args.embed_d, dtype=torch.float32)
        self.a_net_embed = torch.zeros(args.A_n, args.embed_d, dtype=torch.float32)
        self.v_net_embed = torch.zeros(args.V_n, args.embed_d, dtype=torch.float32) 
        
        self.p_a_net_embed = torch.empty(0)
        self.p_v_net_embed = torch.empty(0)
        self.p_p_net_embed = torch.empty(0)
        
        self.a_text_embed = torch.zeros(args.A_n, args.embed_d, dtype=torch.float32)
        self.v_text_embed = torch.zeros(args.V_n, args.embed_d, dtype=torch.float32)
        
        self.p_a_list = torch.empty(0)
        self.p_p_list = torch.empty(0)
        self.p_v_list = torch.empty(0)
        
        self.a_a_edge_index = torch.empty(0)
        self.a_p_edge_index = torch.empty(0)
        self.a_v_edge_index = torch.empty(0) 
        self.v_a_edge_index = torch.empty(0)
        self.v_p_edge_index = torch.empty(0)
        self.v_v_edge_index = torch.empty(0)
        self.p_a_edge_index = torch.empty(0)
        self.p_p_edge_index = torch.empty(0)
        self.p_v_edge_index = torch.empty(0)   
        
        self.a_class = torch.full((args.A_n,), -1, dtype=torch.long)
    
    def read_content_file(self): # p_text, p_abstract, p_net, a_net, v_net

        with open(self.args.data_path, 'r') as file:            
            lines = file.readlines()[1:]       
			
        for i, line in enumerate(lines):
                entries = line.strip().split()
                if f_name == 'p_title_embed.txt':
                    self.p_title_embed[i] = torch.tensor([float(x) for x in entries[1:]]) 
                elif f_name == 'p_abstract_embed.txt':
                    self.p_abstract_embed[i] = torch.tensor([float(x) for x in entries[1:]])
                else:
                    if not i: continue
                    node_type = entries[0][0]
                    node_id = int(entries[0][1:])
                    if node_type == 'a':
                        self.a_net_embed[node_id-1] = torch.tensor([float(x) for x in entries[1:]])
                    elif node_type == 'p':
                        self.p_net_embed[node_id] = torch.tensor([float(x) for x in entries[1:]])
                    elif node_type == 'v':
                        self.v_net_embed[node_id] = torch.tensor([float(x) for x in entries[1:]])
                    else: continue
        
    def read_cite_file(self): # p_a, p_p,p_v lists
        p_a_list = []
        p_p_list = [] 
        p_v_list = []
        
        for f_name in self.cite_filename:
            with open(self.args.data_path + f_name, 'r') as file:            
                lines = file.readlines()

            for i, line in enumerate(lines):
                line = line.strip()
                node_id = int(re.split(':', line)[0])
                neigh_list = re.split(',', re.split(':', line)[1])
                
                for neigh in neigh_list:
                    if f_name == 'p_a_list.txt':
                        p_a_list.append(torch.tensor([[node_id,int(neigh)-1]]))
                    elif f_name == 'p_p_list.txt':
                        p_p_list.append(torch.tensor([[node_id,int(neigh)]]))
                    else:
                        p_v_list.append(torch.tensor([[node_id,int(neigh)]]))

        # Concatenate the list of tensors into a single tensor
        self.p_a_list = torch.cat(p_a_list, dim=0).t().contiguous()
        self.p_p_list = torch.cat(p_p_list, dim=0).t().contiguous()
        self.p_v_list= torch.cat(p_v_list, dim=0).t().contiguous()
        
    def pt_aggr_embed(self): #p_a_net, p_v_net, p_p_net, a_text, v_text
        
        self.p_a_net_embed = aggregate(self.a_net_embed, self.p_a_list, self.args.P_n)
        self.p_v_net_embed = aggregate(self.v_net_embed, self.p_v_list, self.args.P_n)
        self.p_p_net_embed = aggregate(self.p_net_embed, self.p_p_list, self.args.P_n)
        
        for f_name in self.neigh_filename:
            with open(self.args.data_path + f_name, 'r') as file:            
                lines = file.readlines()
                
            for i, line in enumerate(lines):
                line = line.strip()
                node_id = int(re.split(':', line)[0])
                neigh_list = list(map(int, re.split(',', re.split(':', line)[1])))
                
                if f_name == 'a_p_list.txt':
                    if len(neigh_list) >= 3:
                        self.a_text_embed[node_id-1] = torch.mean(self.p_abstract_embed[neigh_list[:3]], dim=0)
                    else:
                        self.a_text_embed[node_id-1] = torch.mean(self.p_abstract_embed[neigh_list], dim=0)
                else:
                    if len(neigh_list) >= 5:
                        self.v_text_embed[node_id] = torch.mean(self.p_abstract_embed[neigh_list[:5]], dim=0)
                    else:
                        self.v_text_embed[node_id] = torch.mean(self.p_abstract_embed[neigh_list], dim=0)

    
    def read_walk_file(self): # all edges indices based on neighbours in random walks
        
        a_edge_list = []
        p_edge_list = []
        v_edge_list = []
        a_a_edge_index = []
        a_p_edge_index = []
        a_v_edge_index = []
        v_a_edge_index = []
        v_p_edge_index = []
        v_v_edge_index = []
        p_a_edge_index = []
        p_p_edge_index = []
        p_v_edge_index = []
        
        with open(self.args.data_path + "het_neigh_train.txt", 'r') as file:            
            lines = file.readlines()
                
        for i, line in enumerate(lines):
            line = line.strip()
            node_type = re.split(':', line)[0][0]
            node_id = int(re.split(':', line)[0][1:])
            neigh_list = re.split(',', re.split(':', line)[1])
            # print(node_type, neigh_list)
            
            a_edge_list = [node for node in neigh_list if node.startswith('a')]
            p_edge_list = [node for node in neigh_list if node.startswith('p')]
            v_edge_list = [node for node in neigh_list if node.startswith('v')]
            # print(a_edge_list)
            
            # Count the frequency of elements starting with 'a', 'p', and 'v'
            a_counts = Counter(a_edge_list)
            p_counts = Counter(p_edge_list)
            v_counts = Counter(v_edge_list)
            # print(a_counts)
            
            a_edge_list = [node for node, count in a_counts.most_common(10)]
            p_edge_list = [node for node, count in p_counts.most_common(10)]
            v_edge_list = [node for node, count in v_counts.most_common(3)]
            # print(a_edge_list)
            
            if node_type == 'a':
                a_a_edge_index.extend([torch.tensor([[node_id-1, int(node[1:])-1]]) for node in a_edge_list])
                a_p_edge_index.extend([torch.tensor([[node_id-1, int(node[1:])]]) for node in p_edge_list])
                a_v_edge_index.extend([torch.tensor([[node_id-1, int(node[1:])]]) for node in v_edge_list])
            elif node_type == 'v':
                v_a_edge_index.extend([torch.tensor([[node_id, int(node[1:])-1]]) for node in a_edge_list])
                v_p_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in p_edge_list])
                v_v_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in v_edge_list])
            else:
                p_a_edge_index.extend([torch.tensor([[node_id, int(node[1:])-1]]) for node in a_edge_list])
                p_p_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in p_edge_list])
                p_v_edge_index.extend([torch.tensor([[node_id, int(node[1:])]]) for node in v_edge_list])
        
        # a_a_edge_index = sorted(a_a_edge_index, key=lambda x: x[0, 1])
        # v_v_edge_index = sorted(v_v_edge_index, key=lambda x: x[0, 1])
        # p_p_edge_index = sorted(p_p_edge_index, key=lambda x: x[0, 1])
           
        # Concatenate the list of tensors into a single tensor
        self.a_a_edge_index = torch.cat(a_a_edge_index, dim=0).t().contiguous()
        self.a_p_edge_index = torch.cat(a_p_edge_index, dim=0).t().contiguous()
        self.a_v_edge_index = torch.cat(a_v_edge_index, dim=0).t().contiguous()
        self.v_a_edge_index = torch.cat(v_a_edge_index, dim=0).t().contiguous()
        self.v_p_edge_index = torch.cat(v_p_edge_index, dim=0).t().contiguous()
        self.v_v_edge_index = torch.cat(v_v_edge_index, dim=0).t().contiguous()
        self.p_a_edge_index = torch.cat(p_a_edge_index, dim=0).t().contiguous()
        self.p_p_edge_index = torch.cat(p_p_edge_index, dim=0).t().contiguous()
        self.p_v_edge_index = torch.cat(p_v_edge_index, dim=0).t().contiguous()
        
        
    def read_label_file(self):
        
        with open(self.args.data_path + "a_class.txt", 'r') as file:            
            lines = file.readlines()
        for i, line in enumerate(lines):
                entries =  line.strip().split(',')
                self.a_class[int(entries[0])-1] = int(entries[1])
 
       
def input_data(args):
    dataset = Aminer(args)
    dataset.read_content_file()
    dataset.read_cite_file()
    dataset.pt_aggr_embed()
    dataset.read_walk_file()
    dataset.read_label_file()

    data = HeteroData()
    
    data['a'].num_nodes = args.A_n
    data['p'].num_nodes = args.P_n
    data['v'].num_nodes = args.V_n

    data['p_title_embed'].x = dataset.p_title_embed
    data['p_abstract_embed'].x = dataset.p_abstract_embed
    data['p_net_embed'].x = dataset.p_net_embed
    data['p_a_net_embed'].x = dataset.p_a_net_embed
    data['p_p_net_embed'].x = dataset.p_p_net_embed
    data['p_v_net_embed'].x = dataset.p_v_net_embed
        
    data['a_net_embed'].x = dataset.a_net_embed
    data['a_text_embed'].x = dataset.a_text_embed
        
    data['v_net_embed'].x = dataset.v_net_embed
    data['v_text_embed'].x = dataset.v_text_embed
    
    data['a'].y = dataset.a_class

    data['a', 'walk', 'a'].edge_index = dataset.a_a_edge_index
    data['a', 'walk', 'p'].edge_index = dataset.a_p_edge_index
    data['a', 'walk', 'v'].edge_index = dataset.a_v_edge_index
    
    data['p', 'walk', 'a'].edge_index = dataset.p_a_edge_index
    data['p', 'walk', 'p'].edge_index = dataset.p_p_edge_index
    data['p', 'walk', 'v'].edge_index = dataset.p_v_edge_index
    
    data['v', 'walk', 'a'].edge_index = dataset.v_a_edge_index
    data['v', 'walk', 'p'].edge_index = dataset.v_p_edge_index
    data['v', 'walk', 'v'].edge_index = dataset.v_v_edge_index
    
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