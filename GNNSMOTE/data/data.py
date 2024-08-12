import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class Cora:
    
    def __init__(self, data_dir):                                  # Directory path (data folder) as parameter
        
        self.data_dir = data_dir
        self.content_file = os.path.join(data_dir, 'cora.content') # Link the content file path to content_file variable
        self.cites_file = os.path.join(data_dir, 'cora.cites')     # Link the cite file path to cite_file variable
        self.node = {}
        
    def read_content_file(self):                                   # For extracting data from content file
        
        with open(self.content_file, 'r') as file:                 # Open the file from path in read only mode 'r'
            lines = file.readlines()                               # lines is a list storing each line as string
            # print(type(file), "\n", type(lines))
            
        num_nodes = len(lines)                                 
        num_features = len(lines[0].split()) - 2                   # split splits each substring  separated by whitespace into a list
        
        x = torch.zeros(num_nodes, num_features, dtype = torch.float32)  # x stores the feature matrix
        y = torch.zeros(num_nodes, dtype = torch.long)                   # y stores the label matrix
        
        # Class dictionary identifies each label with an integer to add to long type tensor y
        class_dict = {'Neural_Networks': 0, 'Reinforcement_Learning': 1, 'Probabilistic_Methods': 2,  
                         'Case_Based': 3, 'Theory': 4, 'Rule_Learning': 5, 'Genetic_Algorithms': 6} 
        
        sorted_lines, self.node = arrange(lines)                       
        
        for i, line in enumerate(sorted_lines):
            entries = line.strip().split()                              # Adding each trimmed line to entries
            if len(entries) < num_features + 2: entries.insert(-2, "0") # Filling in the missing features
            x[i] = torch.tensor([float(x) for x in entries[1:-1]])      # Splitting the first and last element and typecasting to float. Add to x
            y[i] = class_dict[entries[-1]]                              # Add the corresponding integer for label to y from the dictionary
            
        return {'x': x, 'y': y}                           # Return a dictionary with x and y
            
    def read_cite_file(self):                             # For extracting data from cite file
        
        with open(self.cites_file) as file:
            lines = file.readlines()
        
        edge_index = torch.zeros(len(lines), 2, dtype = torch.long)    # Initialising edge_index tensor
            
        for i, line in enumerate(lines):
            entries = [self.node[int(x)] for x in line.strip().split()] # Replace each node with their index from the dictionary
            edge_index[i] = torch.tensor(entries)                       # to integer in a map object for later appending to edge_index
        
        return edge_index.t().contiguous()          # Convert edge_index to transpose form and to contiguous (previous) memory
    
    def load_data(self):                                 # Loads the tensorial data of content and cite files
        
        content = self.read_content_file()
        edge_index = self.read_cite_file()
        
        data = Data(x = content['x'], edge_index = edge_index, y = content['y'])
        
        return data
            
                   
        

def arrange(lines):
    # Extract and sort the nodes based on their original order
    nodes = sorted(set(int(line.split()[0]) for line in lines))
    # Create a dictionary that maps nodes to sequential numbers
    node_mapping = {node: i for i, node in enumerate(nodes)}
    # Sort the lines based on the node mapping
    sorted_lines = sorted(lines, key=lambda line: node_mapping[int(line.split()[0])])  
    return sorted_lines, node_mapping
    
        
# class Dataloader():   
    
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.content, self.edge_index = CoraDataset(data_dir).load_data()
         
#     def train_dataloader(self):
#         train_data = Data(x = self.content['x'][:1000], edge_index = self.edge_index, y = self.content['y'][:1000]) # Convert graph data to standard form after splicing
#         return DataLoader(train_data, batch_size=32, shuffle=True)
    
#     def test_dataloader(self):
#         test_data = Data(x = self.content['x'][1000:], edge_index = self.edge_index, y = self.content['y'][1000:])
#         return DataLoader(test_data, batch_size=32, shuffle=True)
    