import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from smote import smote
from metric import edge_loss, accuracy, evaluate_class_performance

def adj_gen(adj_new, adj_old, ori_num):
    threshold = 0.5
    modified_adj = adj_new.clone()  # Create a copy of adj_new to avoid inplace modification
    modified_adj[modified_adj < threshold] = 0.0
    modified_adj[modified_adj >= threshold] = 1.0
    modified_adj[:ori_num, :ori_num] = adj_old
    return modified_adj


# Train Function on the entire data
def train_smote(data, encoder, classifier, decoder, num_epochs, lr, weight_decay, 
                 train_idx, val_idx, portion, im_class_num, mode):
    
    criterion = nn.CrossEntropyLoss()
    loss_de, edge_ac = 0, 0
    
    epochs = [1, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    val_acc_list = []
    val_auc_list = []
    val_f1_list = []

    # Define your optimizer for encoder and classifier
    optimizer_en = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_de = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    # rec_weight = nn.Parameter(torch.randn(1))
    
    classifier.train()
    decoder.train()
    
    ori_num = data['a'].num_nodes
    adj_old = torch.zeros((ori_num, ori_num), dtype = torch.float32)
    adj_old[data['a', 'walk', 'a'].edge_index[0], data['a', 'walk', 'a'].edge_index[1]] = 1.0

    for epoch in range(num_epochs):

        optimizer_en.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_de.zero_grad()
        
        features = encoder(data)
        
        if mode == 'sm':
            new_features, new_labels , new_train_idx = smote(features = features, labels = data['a'].y, 
                                            train_idx = train_idx, portion = portion, im_class_num = im_class_num)
            adj_new = decoder(new_features, new_features)
            edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj_old, reduction='mean')
            edge_index = adj_gen(adj_new, adj_old, ori_num).nonzero().t().contiguous()
            loss_de = edge_loss(adj_new[:ori_num, :ori_num], adj_old)
        else:
            new_features, new_labels, new_train_idx = features, data['a'].y, train_idx
            edge_index = data['a', 'walk', 'a'].edge_index  
            
        outputs = classifier(new_features, edge_index)
        # print("Finished class", outputs.shape)
        
        loss_cls = criterion(outputs[new_train_idx], new_labels[new_train_idx])
        loss = loss_de * 0.000001 + loss_cls
        # print("Finished loss")
        
        loss.backward()
        optimizer_en.step()
        optimizer_cls.step()
        optimizer_de.step()

        acc = accuracy(outputs[new_train_idx], new_labels[new_train_idx])
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, Edge Accuracy: {(1-edge_ac):.4f}')
        evaluate_class_performance(outputs[new_train_idx], new_labels[new_train_idx])
        
        if val_idx is not None:
            ac, auc, f1 = test_smote(data, encoder, classifier, decoder, val_idx, adj_old,
                       dataset = 'Validation', mode = mode)  # Evaluate on the validation dataset
            
            if epoch in epochs:
                val_acc_list.append(ac)
                val_auc_list.append(auc)
                val_f1_list.append(f1)

    print("Finished Training")
    print("Val_acc_list: ", val_acc_list, "\nVal_auc_list: ", val_auc_list, "\nVal_f1_list: ", val_f1_list)


def test_smote(data, encoder, classifier, decoder, test_idx, adj_old, mode, dataset = "Test"):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    ori_num = data['a'].num_nodes
    loss_de, edge_ac = 0, 0
    

    with torch.no_grad():
        features = encoder(data)
        
        if mode == 'sm':
            adj_new = decoder(features, features)
            edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj_old, reduction='mean')
            edge_index = adj_gen(adj_new, adj_old, ori_num).nonzero().t().contiguous()
            loss_de = edge_loss(adj_new[:ori_num, :ori_num], adj_old)
        else:
            edge_index = data['a', 'walk', 'a'].edge_index

        outputs = classifier(features, edge_index)

        # Compute the validation loss and accuracy
        criterion = nn.CrossEntropyLoss()
        loss_cls = criterion(outputs[test_idx], data['a'].y[test_idx])
        loss = loss_de * 0.000001 + loss_cls
        acc = accuracy(outputs[test_idx], data['a'].y[test_idx])
        
        print(f'{dataset} Loss: {loss.item():.4f}, {dataset} Accuracy: {acc:.4f}, {dataset} Edge Accuracy: {(1-edge_ac):.4f}')
        auc, f1 = evaluate_class_performance(outputs, data['a'].y)
        
    return acc, auc, f1




# Train Function on the entire data
def preD_train(data, encoder, classifier, decoder, num_epochs, lr, weight_decay, 
                 train_idx, val_idx, portion, im_class_num, mode):
    
    criterion = nn.CrossEntropyLoss()
    # loss_de, edge_ac = 0, 0
    
    epochs = [1, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    val_acc_list = []
    val_auc_list = []
    val_f1_list = []

    # Define your optimizer for encoder and classifier
    optimizer_en = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer_de = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    # rec_weight = nn.Parameter(torch.randn(1))
    
    classifier.train()
    decoder.eval()
    
    ori_num = data['a'].num_nodes
    adj_old = torch.zeros((ori_num, ori_num), dtype = torch.float32)
    adj_old[data['a', 'walk', 'a'].edge_index[0], data['a', 'walk', 'a'].edge_index[1]] = 1.0

    for epoch in range(num_epochs):

        optimizer_en.zero_grad()
        optimizer_cls.zero_grad()
        # optimizer_de.zero_grad()
        
        features = encoder(data)
        
        if mode == 'sm':
            new_features, new_labels , new_train_idx = smote(features = features, labels = data['a'].y, 
                                            train_idx = train_idx, portion = portion, im_class_num = im_class_num)
            adj_new = decoder(new_features, new_features)
            edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj_old, reduction='mean')
            edge_index = adj_gen(adj_new, adj_old, ori_num).nonzero().t().contiguous()
            # loss_de = edge_loss(adj_new[:ori_num, :ori_num], adj_old)
        else:
            new_features, new_labels, new_train_idx = features, data['a'].y, train_idx
            edge_index = data['a', 'walk', 'a'].edge_index  
            
        outputs = classifier(new_features, edge_index)
        # print("Finished class", outputs.shape)
        
        loss_cls = criterion(outputs[new_train_idx], new_labels[new_train_idx])
        loss = loss_cls
        # print("Finished loss")
        
        loss.backward()
        optimizer_en.step()
        optimizer_cls.step()

        acc = accuracy(outputs[new_train_idx], new_labels[new_train_idx])
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, Edge Accuracy: {(1-edge_ac):.4f}')
        evaluate_class_performance(outputs[new_train_idx], new_labels[new_train_idx])
        
        if val_idx is not None:
            ac, auc, f1 = test_smote(data, encoder, classifier, decoder, val_idx, adj_old,
                       dataset = 'Validation', mode = mode)  # Evaluate on the validation dataset
            
            if epoch in epochs:
                val_acc_list.append(ac)
                val_auc_list.append(auc)
                val_f1_list.append(f1)

    print("Finished Training")
    print("Val_acc_list: ", val_acc_list, "\nVal_auc_list: ", val_auc_list, "\nVal_f1_list: ", val_f1_list)