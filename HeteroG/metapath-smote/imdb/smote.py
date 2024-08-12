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
def oversample(features, labels, train_idx, edge_index, edge_weight, args):  
    
    train_idx = torch.LongTensor(train_idx).to(args.device)       
    new_features = None                    
    sample_idx_list = []
    near_neighbor_list = []

    avg_number = args.class_samp_num[0]

    for i in args.im_class_num:   
        
        sample_idx = train_idx[(labels==i)[train_idx]]    
        
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
                sample_idx = sample_idx[:left_portion]          
                if left_portion == 0: break

            samples = features[sample_idx, :]    
            
            distance = squareform(pdist(samples.cpu().detach()))  
            np.fill_diagonal(distance,distance.max() + 100)       
            near_neighbor = distance.argmin(axis = -1)           
            interp_place = random.random()                        
            new_samples = samples + (samples[near_neighbor,:] - samples)*interp_place 
            
            if new_features is None: new_features = new_samples
            else:  new_features = torch.cat((new_features, new_samples),0)  
            
            sample_idx_list.extend(sample_idx)  
            near_neighbor_list.extend(near_neighbor)
        
        if new_features is not None:   
            new_labels = labels.new(torch.Size((new_features.shape[0],1))).reshape(-1).fill_(i)
            new_idx = np.arange(features.shape[0], features.shape[0] + new_features.shape[0])
            new_train_idx = train_idx.new(new_idx)                                            
        
            features = torch.cat((features , new_features), 0)      
            labels = torch.cat((labels, new_labels), 0)
            train_idx = torch.cat((train_idx, new_train_idx), 0)

            new_features = None
    
    edge_weight = edge_weight_gen(sample_idx_list, near_neighbor_list, edge_index, edge_weight, args)      
    edge_index = sm_edge_gen(sample_idx_list, near_neighbor_list, edge_index, args)
    
    return features, labels, train_idx, edge_index, edge_weight


def sm_edge_gen(sample_idx_list, near_neighbors, edge_index, args):     
    add_num = len(sample_idx_list)   

    adj = torch.zeros((args.M_n, args.M_n), dtype = torch.float32).to(args.device)
    adj[edge_index[0], edge_index[1]] = 1.0
    
    new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[1]+add_num)))
    
    adj_new = adj.new(torch.clamp_(adj[:,sample_idx_list] + adj[:,near_neighbors], min=0.0, max = 1.0))
    new_adj[:adj.shape[0], adj.shape[1]:] = adj_new[:,:]
    
    adj_new = adj.new(torch.clamp_(adj[sample_idx_list,:][:,sample_idx_list] + adj[near_neighbors,:][:,near_neighbors], min=0.0, max = 1.0))
    new_adj[adj.shape[0]:, adj.shape[1]:] = adj_new[:,:]
    
    adj_new = adj.new(torch.clamp_(adj[sample_idx_list,:] + adj[near_neighbors,:], min=0.0, max = 1.0))
    new_adj[:adj.shape[0], :adj.shape[1]] = adj[:,:]
    
    new_adj[adj.shape[0]:, :adj.shape[1]] = adj_new[:,:]
    
    return new_adj.nonzero().t().contiguous().to(args.device)


def edge_weight_gen(sample_idx_list, near_neighbors, edge_index, edge_weight, args):     
    add_num = len(sample_idx_list)   

    adjw = torch.zeros((args.M_n, args.M_n), dtype = torch.float32).to(args.device)
    adjw[edge_index[0], edge_index[1]] = edge_weight
    
    new_adjw = adjw.new(torch.Size((adjw.shape[0]+add_num, adjw.shape[1]+add_num)))
    
    adjw_new = adjw.new((adjw[:,sample_idx_list] + adjw[:,near_neighbors])/2)
    new_adjw[:adjw.shape[0], adjw.shape[1]:] = adjw_new[:,:]
    
    adjw_new = adjw.new((adjw[sample_idx_list,:][:,sample_idx_list] + adjw[near_neighbors,:][:,near_neighbors])/2)
    new_adjw[adjw.shape[0]:, adjw.shape[1]:] = adjw_new[:,:]
    
    adjw_new = adjw.new((adjw[sample_idx_list,:] + adjw[near_neighbors,:])/2)
    new_adjw[:adjw.shape[0], :adjw.shape[1]] = adjw[:,:]
    
    new_adjw[adjw.shape[0]:, :adjw.shape[1]] = adjw_new[:,:]
    
    return new_adjw[new_adjw.nonzero(as_tuple=True)].view(-1, 1).squeeze().to(args.device)