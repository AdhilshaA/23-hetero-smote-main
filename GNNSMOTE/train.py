import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from smote import smote, smote2
from metric import edge_loss, accuracy, evaluate_class_performance

# Training Function without smote
def train_graph(train_data, val_data, encoder, classifier, num_epochs, lr, weight_decay):
    # Define your loss function (e.g., CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()

    # Define your optimizer for encoder and classifier
    optimizer_en = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    # Set the encoder and classifier in training mode
    encoder.train()
    classifier.train()

    for epoch in range(num_epochs):

        optimizer_en.zero_grad()
        optimizer_cls.zero_grad()

        features = encoder(train_data)
        outputs = classifier(features, train_data.edge_index)

        loss = criterion(outputs,train_data.y)

        loss.backward()
        optimizer_en.step()
        optimizer_cls.step()
        
        acc = accuracy(outputs,train_data.y)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')
        evaluate_class_performance(outputs,train_data.y)
        
        if val_data is not None:
            test_graph(val_data, encoder, classifier, dataset = 'Validation')  # Evaluate on the validation dataset

    print("Finished Training")
    

    
def test_graph(data, encoder, classifier, dataset = "Test"):

    # Set the encoder and classifier in evaluation mode
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        features = encoder(data)
        outputs = classifier(features, data.edge_index)

        # Define your loss function (e.g., CrossEntropyLoss) for validation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, data.y)

        acc = accuracy(outputs, data.y)
        print(f'{dataset} Loss: {loss.item():.4f}, {dataset} Accuracy: {acc:.4f}')
        evaluate_class_performance(outputs, data.y)



# Train Function with smote
def train_smote(train_data, val_data, encoder, classifier, decoder, num_epochs, lr, weight_decay, train_idx, portion, im_class_num):
    torch.autograd.set_detect_anomaly(True)
    criterion = nn.CrossEntropyLoss()

    # Define your optimizer for encoder and classifier
    optimizer_en = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_de = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    
    classifier.train()
    decoder.train()
    
    ori_num = len(train_idx)
    adj_old = torch.zeros((ori_num, ori_num), dtype = torch.float32)
    adj_old[train_data.edge_index[0], train_data.edge_index[1]] = 1.0

    for epoch in range(num_epochs):

        optimizer_en.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_de.zero_grad()
        
        features = encoder(train_data) 
        new_features, new_labels,_ = smote2(features = features, labels = train_data.y, portion = portion, im_class_num = im_class_num)
        
        adj_new = decoder(new_features)
        edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj_old, reduction='mean')
        edge_index = adj_gen(adj_new, adj_old, ori_num).nonzero().t().contiguous()
        
        outputs = classifier(new_features, edge_index)
        
        loss_de = edge_loss(adj_new[:ori_num, :ori_num], adj_old)
        loss_cls = criterion(outputs, new_labels)
        loss = loss_de * 0.000001 + loss_cls

        loss.backward()
        optimizer_en.step()
        optimizer_cls.step()
        optimizer_de.step()

        acc = accuracy(outputs, new_labels)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, Edge Accuracy: {(1-edge_ac):.4f}')
        evaluate_class_performance(outputs, new_labels)
        
        if val_data is not None:
            test_smote(val_data, encoder, classifier, decoder, dataset = 'Validation')  # Evaluate on the validation dataset
    
    torch.autograd.set_detect_anomaly(False)
    print("Finished Training")


def test_smote(data, encoder, classifier, decoder, dataset = "Test"):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    
    ori_num = data.x.shape[0]
    adj_old = torch.zeros((ori_num, ori_num), dtype = torch.float32)
    adj_old[data.edge_index[0], data.edge_index[1]] = 1.0

    with torch.no_grad():
        features = encoder(data)
        adj_new = decoder(features)
        edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj_old, reduction='mean')
        edge_index = adj_gen(adj_new, adj_old, ori_num).nonzero().t().contiguous()

        outputs = classifier(features, edge_index)

        # Compute the validation loss and accuracy
        criterion = nn.CrossEntropyLoss()
        loss_de = edge_loss(adj_new[:ori_num, :ori_num], adj_old)
        loss_cls = criterion(outputs, data.y)
        loss = loss_de + loss_cls
        acc = accuracy(outputs, data.y)

        print(f'{dataset} Loss: {loss.item():.4f}, {dataset} Accuracy: {acc:.4f}, {dataset} Edge Accuracy: {(1-edge_ac):.4f}')
        evaluate_class_performance(outputs, data.y)   
        
        

def adj_gen(adj_new, adj_old, ori_num):
    threshold = 0.5
    modified_adj = adj_new.clone()  # Create a copy of adj_new to avoid inplace modification
    modified_adj[modified_adj < threshold] = 0.0
    modified_adj[modified_adj >= threshold] = 1.0
    modified_adj[:ori_num, :ori_num] = adj_old
    return modified_adj




# Train Function on the entire data
def train_smote2(data, encoder, classifier, decoder, num_epochs, lr, weight_decay, train_idx, val_idx, portion, im_class_num, mode):
    
    criterion = nn.CrossEntropyLoss()
    loss_de = 0
    edge_ac = 0

    # Define your optimizer for encoder and classifier
    optimizer_en = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_de = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    # rec_weight = nn.Parameter(torch.randn(1))
    
    classifier.train()
    decoder.train()
    
    ori_num = data.x.shape[0]
    adj_old = torch.zeros((ori_num, ori_num), dtype = torch.float32)
    adj_old[data.edge_index[0], data.edge_index[1]] = 1.0

    for epoch in range(num_epochs):

        optimizer_en.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_de.zero_grad()
        
        features = encoder(data) 
        
        if mode == "sm":
            new_features, new_labels , new_train_idx = smote(features = features, labels = data.y, 
                                            train_idx = train_idx, portion = portion, im_class_num = im_class_num)
        
            adj_new = decoder(new_features)
            edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj_old, reduction='mean')
            edge_index = adj_gen(adj_new, adj_old, ori_num).nonzero().t().contiguous()
            loss_de = edge_loss(adj_new[:ori_num, :ori_num], adj_old)
        else:
            new_features, new_train_idx, new_labels = features, train_idx, data.y
            edge_index = data.edge_index
        
        outputs = classifier(new_features, edge_index)
        
        loss_cls = criterion(outputs[new_train_idx], new_labels[new_train_idx])
        loss = loss_de * 0.000001 + loss_cls

        loss.backward()
        optimizer_en.step()
        optimizer_cls.step()
        optimizer_de.step()

        acc = accuracy(outputs[new_train_idx], new_labels[new_train_idx])
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, Edge Accuracy: {(1-edge_ac):.4f}')
        evaluate_class_performance(outputs[new_train_idx], new_labels[new_train_idx])
        
        if val_idx is not None:
            test_smote2(data, encoder, classifier, decoder, val_idx, dataset = 'Validation', mode = mode)  # Evaluate on the validation dataset

    print("Finished Training")


def test_smote2(data, encoder, classifier, decoder, test_idx, mode, dataset = "Test"):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    loss_de = 0
    edge_ac = 0
    
    ori_num = data.x.shape[0]
    adj_old = torch.zeros((ori_num, ori_num), dtype = torch.float32)
    adj_old[data.edge_index[0], data.edge_index[1]] = 1.0

    with torch.no_grad():
        features = encoder(data)
        adj_new = decoder(features)
        
        if mode == "sm":
            edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj_old, reduction='mean')
            edge_index = adj_gen(adj_new, adj_old, ori_num).nonzero().t().contiguous()
            loss_de = edge_loss(adj_new[:ori_num, :ori_num], adj_old)
        else:
            edge_index = data.edge_index

        outputs = classifier(features, edge_index)

        # Compute the validation loss and accuracy
        criterion = nn.CrossEntropyLoss()
        loss_cls = criterion(outputs[test_idx], data.y[test_idx])
        loss = loss_de * 0.000001 + loss_cls
        acc = accuracy(outputs[test_idx], data.y[test_idx])

        print(f'{dataset} Loss: {loss.item():.4f}, {dataset} Accuracy: {acc:.4f}, {dataset} Edge Accuracy: {(1-edge_ac):.4f}')
        evaluate_class_performance(outputs, data.y)   