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

def oversample(features, labels, target_edge, train_idx, edge_indices_list, args, os_mode):  
    
    train_idx = torch.LongTensor(train_idx).to(args.device)      
    new_features = [None, None]           
    embed_edge = None         
    sample_idx_list = [[], []]
    near_neighbor_list = [[], []]
    new_samples = [None, None]      
    edge_num = target_edge.shape[0]

    avg_number = args.class_samp_num[0]

    for i in args.im_class_num: 

        sample_idx = train_idx[(labels==i)[train_idx]]    # Stores the indices from train_idx that belong to class i
        
        if args.portion == 0: 
            c_portion = int(avg_number/sample_idx.shape[0])   
            portion_rest = (avg_number/sample_idx.shape[0]) - c_portion 

        else:
            c_portion = int(args.portion)
            portion_rest = args.portion - c_portion
        
        for j in range(c_portion): 
          
            if j == c_portion - 1: 
                if portion_rest == 0.0: break
                left_portion = round(sample_idx.shape[0]*portion_rest) 
                if left_portion == 0: break
                sample_idx = sample_idx[:left_portion]          

            parent_edges = target_edge[sample_idx, :]           # Picks out the sample_idx target edges
            source = features[0][parent_edges[:, 0]]            # Extract embeddings based on parent_labels
            target = features[1][parent_edges[:, 1]]
            edge_embed = source * target                  # Perform element-wise multiplication     
            
            if os_mode == 'up':
                new_samples[0] = source
                new_samples[1] = target
                    
            else:
                distance = squareform(pdist(edge_embed.cpu().detach()))  # Calculates distance between each pair of nodes into square symmetric matrix A[i,j] = Distance between i-th and j-th node
                np.fill_diagonal(distance,distance.max() + 10000)         # Make the diagonal element which would be 0 as maximum since we can't interpolate between the same node
                near_neighbor = distance.argmin(axis = -1)               # Stores the indices of the nearest neighbour in each row for each node. These pairs of nodes will be interpolated on.
                interp_place = random.random()                           # Define weight for interpolation
                
                if os_mode == 'edge_sm':
                    new_edge_embed = edge_embed + (edge_embed[near_neighbor,:] - edge_embed)*interp_place 
                    
                    if embed_edge is None: embed_edge = edge_embed
                    else: embed_edge = torch.cat((embed_edge, new_edge_embed), dim = 0)
                    
                    labels = torch.cat((labels, labels.new(torch.Size((new_edge_embed.size(0),1))).reshape(-1).fill_(i).to(labels.device)), 0)
                    
                    new_idx = train_idx.new(np.arange(edge_num, edge_num + sample_idx.shape[0]))
                    train_idx = torch.cat((train_idx, new_idx),0)
                    edge_num += sample_idx.shape[0] 
                    
                else:   
                    new_samples[0] = source + (source[near_neighbor,:] - source)*interp_place 
                    new_samples[1] = target + (target[near_neighbor,:] - target)*interp_place
            
            if os_mode != 'edge_sm':
                if new_features[0] is None: 
                    new_features[0] = new_samples[0]
                    new_features[1] = new_samples[1]
                else:  
                    new_features[0] = torch.cat((new_features[0], new_samples[0]),0)  
                    new_features[1] = torch.cat((new_features[1], new_samples[1]),0) 
                
                sample_idx_list[0].extend(parent_edges[:,0].flatten())  
                sample_idx_list[1].extend(parent_edges[:,1].flatten()) 
                
                if os_mode != 'up':          
                    near_neighbor_list[0].extend(parent_edges[near_neighbor,:][:,0].flatten())
                    near_neighbor_list[1].extend(parent_edges[near_neighbor,:][:,1].flatten())
        
        if new_features[0] is not None and os_mode != 'edge_sm':   
            new_labels = labels.new(torch.Size((new_features[0].size(0),1))).reshape(-1).fill_(i)
            
            new_edge_idx = torch.tensor([[features[0].size(0)+i,features[1].size(0)+i] for i in range(0,new_features[0].size(0))]).to(labels.device)
            
            new_idx = np.arange(target_edge.shape[0], target_edge.shape[0] + new_labels.shape[0])
            new_train_idx = train_idx.new(new_idx)  
            train_idx = torch.cat((train_idx, new_train_idx), 0)
            
            target_edge = torch.cat((target_edge, new_edge_idx), dim =0)
            features[0] = torch.cat((features[0], new_features[0].to(target_edge.device)), 0)     
            features[1] = torch.cat((features[1], new_features[1].to(target_edge.device)), 0)  
            labels = torch.cat((labels, new_labels.to(target_edge.device)), 0)
            new_features = [None, None]
    
    
    if edge_indices_list is not None:
        edges_list = []
        
        for i in range(2): 
            if os_mode == 'smote' or os_mode == 'gsm':
                edge_list = sm_edge_gen(sample_idx_list, near_neighbor_list, edge_indices_list[i], args, k=i)
            else: 
                edge_list = up_edge_gen(sample_idx_list, edge_indices_list[i], args, k=i)

            edges_list.append(edge_list)
        return features, labels, train_idx, edges_list, target_edge
        
    elif os_mode == 'edge_sm':
        ori_edge_em = features[0][target_edge[:, 0]] * features[1][target_edge[:, 1]] 
        # new_labels = labels.new(torch.Size((embed_edge.size(0),1))).reshape(-1).fill_(i).to(labels.device)
        embed_edge = torch.cat((ori_edge_em, embed_edge),0)
        # labels = torch.cat((labels, new_labels), 0)
        return embed_edge, labels, train_idx
        
    else: return features, labels, train_idx, target_edge


        

def up_edge_gen(sample_idx_list, edge_indices, args, k):     
    edge_list = []       
    add_num = len(sample_idx_list[0])
    
    for i in range(len(edge_indices)):
        
        adj = torch.zeros((args.node_dim[k], args.node_dim[i]), dtype = torch.float32).to(args.device)
        adj[edge_indices[i][0], edge_indices[i][1]] = 1.0
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1]+add_num)))
        
        new_adj[:adj.shape[0], :adj.shape[1]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[1]] = adj[sample_idx_list[k],:]
        new_adj[:adj.shape[0], adj.shape[1]:] = adj[:,sample_idx_list[i]]
        new_adj[adj.shape[0]:, adj.shape[1]:] = adj[sample_idx_list[k],:][:,sample_idx_list[i]]
        
        edge_list.append(new_adj.nonzero().t().contiguous().to(args.device))
        
    return edge_list


def sm_edge_gen(sample_idx_list, near_neighbors, edge_indices, args, k):     
    edge_list = []       
    add_num = len(sample_idx_list[0])   
    
    for i in range(len(edge_indices)):
        
        adj = torch.zeros((args.node_dim[k], args.node_dim[i]), dtype = torch.float32).to(args.device)
        # print(k, i, adj.shape, sample_idx_list[k], "\n", near_neighbors[k])
        adj[edge_indices[i][0], edge_indices[i][1]] = 1.0
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1]+add_num)))
        
        new_adj[:adj.shape[0], :adj.shape[1]] = adj[:,:]
        
        adj_new = adj.new(torch.clamp_(adj[sample_idx_list[k],:] + adj[near_neighbors[k],:], min=0.0, max = 1.0))
        new_adj[adj.shape[0]:, :adj.shape[1]] = adj_new[:,:]
        
        adj_new = adj.new(torch.clamp_(adj[:,sample_idx_list[i]] + adj[:,near_neighbors[i]], min=0.0, max = 1.0))
        new_adj[:adj.shape[0], adj.shape[1]:] = adj_new[:,:]
        
        adj_new = adj.new(torch.clamp_(adj[sample_idx_list[k],:][:,sample_idx_list[i]] + adj[near_neighbors[k],:][:,near_neighbors[i]], min=0.0, max = 1.0))
        new_adj[adj.shape[0]:, adj.shape[1]:] = adj_new[:,:]
    
        edge_list.append(new_adj.nonzero().t().contiguous().to(args.device))
        
    return edge_list
