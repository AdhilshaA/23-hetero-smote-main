import torch
import torch.nn as nn 
import numpy as np
import random
from scipy.spatial.distance import pdist,squareform

'''
train_idx: Indices of the nodes used for training
portion: Number of times the nodes of a class is to be sampled
im_class_num: The upper cap class number of the imbalanced classes
os_mode: oversampling type
'''

def oversample(features, labels, target_edge, train_idx, edge_indices, args, os_mode):  
    
    train_idx = torch.LongTensor(train_idx).to(args.device)
    max_class = 1        
    new_features = None            
    embed_edge = None         
    sample_idx_list = []
    near_neighbor_list = []

    avg_number = int(train_idx.shape[0]/(max_class+1))   # Stores the average number of nodes per class

    for i in args.im_class_num:    # Loop interates over all imbalanced classes to apply smote on

        sample_idx = train_idx[(labels==i)[train_idx]]    # Stores the indices from train_idx that belong to class i
        
        if args.portion == 0: #refers to even distribution
            c_portion = int(avg_number/sample_idx.shape[0])   # c_portion holds the integer times the sample_idx are to be sampled to reach average number of nodes
            portion_rest = (avg_number/sample_idx.shape[0]) - c_portion # Stores the leftover fraction of the sampling to be done

        else:
            c_portion = int(args.portion)
            portion_rest = args.portion - c_portion
        
        for j in range(c_portion):  # This loops runs about the number of times of upsampling of all nodes + fractional
          
            if j == c_portion - 1: # When only the leftover fractional part of total sampling is left
                if portion_rest == 0.0: break
                left_portion = int(sample_idx.shape[0]*portion_rest)  # stores the number of indices to be sampled
                sample_idx = sample_idx[:left_portion]          # Stores the subset of indices to be sampled

            parent_edges = target_edge[sample_idx, :]           # Picks out the sample_idx target edges
            source = features[parent_edges[:, 0]]               # Extract embeddings based on parent_labels
            target = features[parent_edges[:, 1]]
            edge_embed = source * target                   # Perform element-wise multiplication     
            
            if os_mode == 'up':
                new_source =  source
                new_target = target
                new_samples = torch.empty(2 * source.size(0), source.size(1))
                new_samples[0::2] = new_source
                new_samples[1::2] = new_target
                    
            else:
                distance = squareform(pdist(edge_embed.cpu().detach()))  # Calculates distance between each pair of nodes into square symmetric matrix A[i,j] = Distance between i-th and j-th node
                np.fill_diagonal(distance,distance.max() + 100)          # Make the diagonal element which would be 0 as maximum since we can't interpolate between the same node
                near_neighbor = distance.argmin(axis = -1)               # Stores the indices of the nearest neighbour in each row for each node. These pairs of nodes will be interpolated on.
                interp_place = random.random()                           # Define weight for interpolation
                
                if os_mode == 'edge_sm':
                    new_edge_embed = edge_embed + (edge_embed[near_neighbor,:] - edge_embed)*interp_place 
                    
                    if embed_edge is None: embed_edge = edge_embed
                    else: embed_edge = torch.cat((embed_edge, new_edge_embed), dim = 0)
                    
                    new_idx = train_idx.new(np.arange(target_edge.shape[0], target_edge.shape[0] + sample_idx.shape[0]))
                    train_idx = torch.cat((train_idx, new_idx),0)
                    new_samples = None
                    
                else:   
                    new_source = source + (source[near_neighbor,:] - source)*interp_place 
                    new_target = target + (target[near_neighbor,:] - target)*interp_place
                
                    new_samples = torch.empty(2 * source.size(0), source.size(1))
                    new_samples[0::2] = new_source
                    new_samples[1::2] = new_target
            
            if new_features is None: new_features = new_samples
            else:  new_features = torch.cat((new_features, new_samples),0)  
            
            sample_idx_list.extend(parent_edges.flatten())  
            
            if os_mode != 'up':          
                near_neighbor_list.extend(parent_edges[near_neighbor,:].flatten())
        
        if new_features is not None:   
            new_labels = labels.new(torch.Size((int(new_features.shape[0]/2),1))).reshape(-1).fill_(i)
            
            new_feature_idx = torch.tensor(np.arange(features.shape[0], features.shape[0] + new_features.shape[0])).to(target_edge.device) 
            new_idx = np.arange(target_edge.shape[0], target_edge.shape[0] + new_feature_idx.view(-1,2).shape[0])
            target_edge = torch.cat((target_edge, new_feature_idx.view(-1,2)), dim =0)
    
            new_train_idx = train_idx.new(new_idx)   
            train_idx = torch.cat((train_idx, new_train_idx), 0)
                                                     
            features = torch.cat((features , new_features.to(target_edge.device)), 0)      
            labels = torch.cat((labels, new_labels.to(target_edge.device)), 0)
            new_features = None
        
    if edge_indices is not None:
        if os_mode == 'smote' or os_mode == 'gsm':
            edge_list = sm_edge_gen(sample_idx_list, near_neighbor_list, edge_indices, args, k=1)
        else:
            edge_list = up_edge_gen(sample_idx_list, edge_indices, args, k=1)
        return features, labels, train_idx, edge_list, target_edge
    
    elif os_mode == 'edge_sm':
        ori_edge_em = features[target_edge[:, 0]] * features[target_edge[:, 1]] 
        new_labels = labels.new(torch.Size((embed_edge.size(0),1))).reshape(-1).fill_(i).to(features.device)
        embed_edge = torch.cat((ori_edge_em, embed_edge),0)
        labels = torch.cat((labels, new_labels), 0)
        return embed_edge, labels, train_idx
    
    else: return features, labels, train_idx, target_edge


        

def up_edge_gen(sample_idx_list, edge_indices, args, k):     
    edge_list = []       
    add_num = len(sample_idx_list)    
    
    for i in range(len(edge_indices)):
        # print("add_num", add_num)
        adj = torch.zeros((args.node_dim[k], args.node_dim[i]), dtype = torch.float32).to(args.device)
        adj[edge_indices[i][0], edge_indices[i][1]] = 1.0
        if i == k:
            new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1]+add_num)))
            new_adj[:adj.shape[0], adj.shape[1]:] = adj[:,sample_idx_list]
            new_adj[adj.shape[0]:, adj.shape[1]:] = adj[sample_idx_list,:][:,sample_idx_list]
        else:
            new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1])))
        new_adj[:adj.shape[0], :adj.shape[1]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[1]] = adj[sample_idx_list,:]
        edge_list.append(new_adj.nonzero().t().contiguous().to(args.device))
        
    return edge_list


def sm_edge_gen(sample_idx_list, near_neighbors, edge_indices, args, k):     
    edge_list = []       
    add_num = len(sample_idx_list)   
    
    for i in range(len(edge_indices)):
        adj = torch.zeros((args.node_dim[k], args.node_dim[i]), dtype = torch.float32).to(args.device)
        adj[edge_indices[i][0], edge_indices[i][1]] = 1.0
        
        if i == k:
            new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1]+add_num)))
            adj_new = adj.new(torch.clamp_(adj[:,sample_idx_list] + adj[:,near_neighbors], min=0.0, max = 1.0))
            new_adj[:adj.shape[0], adj.shape[1]:] = adj_new[:,:]
            adj_new = adj.new(torch.clamp_(adj[sample_idx_list,:][:,sample_idx_list] + adj[near_neighbors,:][:,near_neighbors], min=0.0, max = 1.0))
            new_adj[adj.shape[0]:, adj.shape[1]:] = adj_new[:,:]
            
        else:
            new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1])))
        
        adj_new = adj.new(torch.clamp_(adj[sample_idx_list,:] + adj[near_neighbors,:], min=0.0, max = 1.0))
        new_adj[:adj.shape[0], :adj.shape[1]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[1]] = adj_new[:,:]
        edge_list.append(new_adj.nonzero().t().contiguous().to(args.device))
        
    return edge_list
