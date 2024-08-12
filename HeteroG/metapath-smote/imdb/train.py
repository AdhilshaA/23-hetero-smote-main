import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
from smote import oversample
from metric import edge_loss, accuracy, evaluate_class_performance

w1 = 4
w2 = 2
w3 = 2

# Train Function on the entire data
def train_smote(data, encoder1, encoder2, classifier, decoder, train_idx, val_idx, test_idx, args, train_mode):
    
    epochs = list(range(0, args.num_epochs, 10))

    val_acc_list, val_auc_list, val_f1_list = [], [], []
    test_acc_list, test_auc_list, test_f1_list = [], [], []
    auc_list, f1_list = [], []
    loss_de, edge_ac, max_acc, max_auc, max_f1 = 0, 0, 0, 0, 0

    # Define your optimizer for encoder and classifier
    optimizer_en1 = optim.Adam(encoder1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_en2 = optim.Adam(encoder2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_de = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    encoder1.train()
    encoder2.train()
    decoder.train()
    classifier.train()
    
    edge_index = data['mp'].edge_index
    edge_weight = data['mp'].edge_weight
    x_list = [data['mp1'].x, data['mp2'].x, data['mp3'].x]
    labels = data['mp'].y
    num_node = data['m'].num_nodes
    
    for epoch in range(args.num_epochs):

        optimizer_en1.zero_grad()
        optimizer_en2.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_de.zero_grad()
        
        x = encoder1(x_list)
        x = encoder2(x, edge_index, num_node, edge_weight)
    
        x, new_labels, new_train_idx, ar_edge_index, ar_edge_weight = oversample(x, labels, train_idx, 
        edge_index, edge_weight, args)

        edge_ac, loss_de, new_edge_index, new_edge_weight = het_decode(decoder, x, edge_index, edge_weight, 
        ar_edge_index, ar_edge_weight, args, train_mode, dataset = 'Train')
        
        outputs = classifier(x, new_edge_index, x.size(0), new_edge_weight)
        
        loss_cls = F.cross_entropy(outputs[new_train_idx], new_labels[new_train_idx])
        
        if train_mode == 'preO' or train_mode == 'preT' or train_mode == 'pret'  or train_mode == 'preo': 
            loss = loss_de*args.de_weight + loss_cls
        elif train_mode == 'recon': loss = loss_de * args.de_weight * 10
        else: loss = loss_cls
        
        loss.backward()

        optimizer_en1.step()
        optimizer_en2.step()
        optimizer_de.step()       
        optimizer_cls.step()
        
        # if epoch in epochs:      
        acc = accuracy(outputs[new_train_idx].detach(), new_labels[new_train_idx].detach())
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, Edge Accuracy: {edge_ac}')
        evaluate_class_performance(outputs[new_train_idx].clone().detach(), new_labels[new_train_idx].clone().detach(), 
         thresh_list = [], dataset = 'Train', args = args)
        
        optimizer_en1.zero_grad()
        optimizer_en2.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_de.zero_grad()
            
        if val_idx is not None:
            ac, auc, f1, auc_scr, f1_scr = test_smote(data, encoder1, encoder2, classifier, decoder, val_idx, args = args, 
            dataset='Validation', train_mode=train_mode)  
            
            if epoch in epochs:
                val_acc_list.append(ac)
                val_auc_list.append(auc)
                val_f1_list.append(f1)
        
        if test_idx is not None:
            ac, auc, f1, auc_scr, f1_scr = test_smote(data, encoder1, encoder2, classifier, decoder, test_idx, args = args, 
            dataset='Test', train_mode=train_mode)  
            
            if max_acc < ac: max_acc = ac
            if max_auc < auc: max_auc = auc
            if max_f1 < f1: max_f1 = f1
            
            if epoch in epochs:
                test_acc_list.append(ac)
                test_auc_list.append(auc)
                test_f1_list.append(f1)
        
            if epoch == 149:
                auc_list = auc_scr
                f1_list = f1_scr
                 
            print()
    print("Finished Training")
    print("Validation Metrics:")
    print("Val_acc_list:", ["{:.4f}".format(val) for val in val_acc_list])
    print("Val_auc_list:", ["{:.4f}".format(val) for val in val_auc_list])
    print("Val_f1_list:", ["{:.4f}".format(val) for val in val_f1_list])
    print("Test Metrics:")
    print("Test_acc_list:", ["{:.4f}".format(val) for val in test_acc_list])
    print("Test_auc_list:", ["{:.4f}".format(val) for val in test_auc_list])
    print("Test_f1_list:", ["{:.4f}".format(val) for val in test_f1_list])
    
    test_acc_list.append(max_acc)
    test_auc_list.append(max_auc)
    test_f1_list.append(max_f1)
    
    return test_acc_list, test_auc_list, test_f1_list, auc_list, f1_list

# Test function
def test_smote(data, encoder1, encoder2, classifier, decoder, test_idx, train_mode, args, dataset = "Test"):

    encoder1.eval()
    encoder2.eval()
    classifier.eval()
    decoder.eval()
    
    loss_de, edge_ac = [], []
    auc_list, f1_list = [], []
    
    if dataset == 'Validation': thresh_list = []
    else: thresh_list = args.best_threshold
    
    edge_index = data['mp'].edge_index
    edge_weight = data['mp'].edge_weight
    x_list = [data['mp1'].x, data['mp2'].x, data['mp3'].x]
    labels = data['mp'].y
    num_node = data['m'].num_nodes

    with torch.no_grad():
        x = encoder1(x_list)
        x = encoder2(x, edge_index, num_node, edge_weight)
    
        edge_ac, loss_de, new_edge_index, new_edge_weight = het_decode(decoder, x, edge_index, edge_weight, 
        ar_edge_index = edge_index, ar_edge_weight = edge_weight, args = args, train_mode = train_mode, dataset = dataset)
        
        outputs = classifier(x, new_edge_index, num_node, new_edge_weight)
        
        loss_cls =  F.cross_entropy(outputs[test_idx], labels[test_idx])
        
        if train_mode == 'preO' or train_mode == 'preT' or train_mode == 'pret'  or train_mode == 'preo': 
            loss = loss_de*args.de_weight + loss_cls
        elif train_mode == 'recon': loss = loss_de * args.de_weight * 10  
        else: loss = loss_cls            
        
        # print(loss)
        acc = accuracy(outputs[test_idx].detach(), labels[test_idx].detach()) 
        print(f'{dataset} Loss: {loss.item():.4f}, {dataset} Accuracy: {acc:.4f}, {dataset} Edge Accuracy: {edge_ac}')
        auc, f1, auc_list, f1_list = evaluate_class_performance(outputs[test_idx].clone().detach(), 
        labels[test_idx].clone().detach(), thresh_list = thresh_list, dataset = dataset, args = args)
        
    return acc, auc, f1, auc_list, f1_list


# Function to evalute all decoder related tasks
def het_decode(decoder, x, edge_index, edge_weight, ar_edge_index, ar_edge_weight, args, train_mode, dataset):
    edge_ac, loss_de = 0, 0
 
    adj_new = decoder(x)
    ori_size = args.M_n
    
    adj_old = torch.zeros((ori_size, ori_size), dtype = torch.float32, device=x.device)
    adj_old[edge_index[0], edge_index[1]] = 1.0

    adj_ar = torch.zeros((x.size(0), x.size(0)), dtype = torch.float32, device=x.device)
    adj_ar[ar_edge_index[0], ar_edge_index[1]] = 1.0
    adjw_ar = torch.zeros((x.size(0), x.size(0)), dtype = torch.float32, device=x.device)
    adjw_ar[ar_edge_index[0], ar_edge_index[1]] = ar_edge_weight

    loss_de = edge_loss(adj_new[:ori_size, :ori_size], adj_old)
    adj_new, adjw_ar, edge_ac = adj_gen(adj_new, adj_old, adj_ar, adjw_ar, ori_size, dataset)
    
    new_edge_index = adj_new.nonzero().t().contiguous()
    new_edge_weight = adjw_ar[adjw_ar.nonzero(as_tuple=True)].view(-1, 1).squeeze()
    
    if train_mode != 'preO' and train_mode != 'preo': new_edge_index = new_edge_index.detach()
            
    return edge_ac, loss_de, new_edge_index, new_edge_weight


def adj_gen(adj_new, adj_old, adj_ar, adjw_ar, ori_size, mode):
    threshold = 0.5
    adj_new = (adj_new >= threshold).float()
    adjw_new = adjw_ar
    
    adj_new = torch.mul(adj_ar, adj_new)

    diff = torch.abs(adj_new[:ori_size, :ori_size] - adj_old)
    correct_edges = (diff == 0).sum().item()
    total_edges = diff.numel()
    acc = correct_edges / total_edges
    
    if mode == 'Train': adj_new[:ori_size, :ori_size] = adj_old
        
    adjw_new = torch.mul(adjw_ar, adj_new)    

    return adj_new, adjw_new, acc


''''

From paper:
0) no- No oversampling, no decoder. Loss = loss_cls
1) upscale- Generate samples by repeating raw data and copy the new adj rows of the synthetic nodes from the parent. Use no decoder. 
Only use classification loss. Loss = loss_cls
2) smote: Do the same thing as upscale but use smote instead of upscale. No decoder. Loss = loss_cls
3) embed-smote: Use smote after encoder. adj_new =adj_old. No decoder. Loss  = loss_cls
4) reweight: No oversampling or decoder. Loss = loss_cls 
---- Use of decoder next (For some reason they eliminated the eliminated the edges between synthetic neighbours) ----
---- In smote, adj_new = decoder(...), adj_up = from parent indices ----
5) recon- Pretrain decoder and encoder together. Loss = loss_de. 
6) newG_cls- Use pretrained encoder and decoder but do not finetune them. Loss = loss_cls
7) recon_newG- Use pretrained encoder and decoder and finetune both. Loss = loss_cls + loss_de * weight
8) opt_new_G- if not, then adj_new is detached .detach() before classification. Which means if we backward propogate 
Loss = loss_cls + loss_de...,  adj and hence the decoder is only updated by loss_de & not by loss_cls which is dependent on detached adj

9) edge_sm: smote on edges
Mine:
0) no
1) upscale 
2) smote
3) embed_sm (Smote, no decoder (adj_new = adj_old), loss = loss_cls)
4) reweight (No smote, decoder, only reweight on loss_cls)
5) recon (Only for training the encoder decoder on loss = loss_de)
6) newG (Encoder, decoder aren't finetuned)
6.5) noFT (Decoder isn't finetuned)
7) preT = no_recon - opt_new_G or recon - opt_new_G  (Decoder is finetuned only loss_de)
8) preO = no_recon + opt_new_G or recon + opt_new_G  (Decoder is finetuned on complete loss)
(5-8) + gsm (graphsmote)
''' 