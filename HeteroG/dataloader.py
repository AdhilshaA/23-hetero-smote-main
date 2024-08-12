import torch
import numpy as np
import random
from torch_geometric.data import Data

def train_num(labels, im_class_num, class_sample_num, im_ratio):
    c_train_num = []
    max_class = labels.max().item()
    for i in range(max_class + 1):
        if i in im_class_num:                                              #only imbalance the last classes
            c_train_num.append(int(class_sample_num * im_ratio[im_class_num.index(i)]))
        else:
            c_train_num.append(class_sample_num)
    return c_train_num
    
    
    
# Function to randomly select the nodes for test, train and val based on the required samples from each class
def segregate(labels, c_train_num, args):

    num_classes = max(labels.tolist())+1    
    c_idx = []                                         # class-wise index: Holds indices of nodes for each class
    train_idx = []                                      # These three lists hold the indices for all classes to be used in respective sets
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)   # Rows signify each class, column 1: Number of samples for training, col 2: val, col 3: test
    c_num_mat[:,1] = args.class_samp_num[1]                                 # Validation will have 25 samples per class, test will have 55 samples per class
    c_num_mat[:,2] = args.class_samp_num[2]
    # print("c_num_mat:", c_num_mat, c_num_mat.shape)
    
    for i in range(num_classes):                        # Looping through all classes, to separate out the desired no. of samples
        # print("For class {}:".format(i))
        c_idx = (labels==i).nonzero()[: ,-1].tolist()    # Filtering out the indices of nodes belonging to class i
        print(i, len(c_idx))
        # print("c_idx:", c_idx, len(c_idx))
        random.shuffle(c_idx)              
        # print("c_idx (shuffled):", c_idx, len(c_idx))
        
        train_idx = train_idx + c_idx[:c_train_num[i]]  # Add the indices for class i to training based on c_train_num
        c_num_mat[i,0] = c_train_num[i]                 # Adding training size info

        val_idx = val_idx + c_idx[c_train_num[i]:c_train_num[i]+args.class_samp_num[1]]          
        test_idx = test_idx + c_idx[c_train_num[i]+args.class_samp_num[1]:c_train_num[i]+args.class_samp_num[1]+args.class_samp_num[2]]

    random.shuffle(train_idx)                           # Shuffling to remove bias

    return train_idx, val_idx, test_idx, c_num_mat



# Function for splitting the desired nodes and edges from original set for downstream task and converting to geometric format
def dataloader(data, node_idx):                                         
                                                                        
    l_subset = data.y[node_idx]                                         # Holds the subset of labels for the downstream task
    f_subset = data.x[node_idx]                                         # Holds the subset of features for the downstream task
    node_dict = {node: k for k, node in enumerate(node_idx)}            # Dictionary marking the node indices with new indices in list
    e_subset = edge_mask(data.edge_index, node_idx)                     # Holds subset of edges with nodes from node_idx
        
    e_subset[0] = torch.tensor([node_dict[node_id.item()] for node_id in e_subset[0]]) # Replace source nodes using the dictionary 
    e_subset[1] = torch.tensor([node_dict[node_id.item()] for node_id in e_subset[1]]) # Replace target nodes using the dictionary
    
    return Data(x = f_subset, edge_index = e_subset, y = l_subset) # Convert into data object
        

# Function for creating a boolean mask to filter rows where both source and target nodes are in node_idx       
def edge_mask(edge_index, node_idx):
   
    mask = torch.zeros(edge_index.size(1), dtype=torch.bool)   
    for i in range(edge_index.size(1)):                                 
        source, target = edge_index[0, i], edge_index[1, i]
        if source in node_idx and target in node_idx:
            mask[i] = True 
    
    filtered_edge_index = edge_index[:, mask]  # Apply the mask to filter the rows of edge_index
    return filtered_edge_index