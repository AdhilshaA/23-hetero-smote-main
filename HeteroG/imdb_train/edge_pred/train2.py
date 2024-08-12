import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
from smote import oversample
from metric import edge_loss, accuracy, evaluate_class_performance
from sklearn import linear_model
from sklearn.metrics import classification_report, roc_auc_score


def train_smote(data, edge_indices, target_edge, encoder, encoder2, decoder_list, train_idx, val_idx, 
                test_idx, args, os_mode, train_mode):
    
    edge_em = None
    encoder.eval()
    encoder2.eval()
    for decoder in decoder_list: decoder.eval()
    thresh = args.thresh_dict[os_mode]
    
    if os_mode == 'reweight': classifier = linear_model.LogisticRegression(class_weight='balanced', max_iter=args.num_epochs)
    else: classifier = linear_model.LogisticRegression(max_iter=args.num_epochs)

    with torch.no_grad():
        
        x_list = encoder(data, os_mode)
        
        if os_mode == 'gsm':
            x_list[1], new_labels , new_train_idx, ar_edge_indices, new_target_edge = oversample(features = x_list[1], labels = data['e'].y, 
            target_edge = target_edge, train_idx = train_idx, edge_indices = edge_indices, args= args, os_mode = 'gsm')
            edge_ac, loss_de, new_edge_indices = het_decode(decoder_list, x_list, edge_indices, ar_edge_indices, args, train_mode, dataset = 'Train')
        elif os_mode == 'embed_sm':
            x_list[1], new_labels , new_train_idx, new_target_edge = oversample(features = x_list[1], labels = data['e'].y, 
            target_edge = target_edge, train_idx = train_idx, edge_indices = None, args= args, os_mode = 'gsm')
            new_edge_indices = edge_indices
        elif os_mode == 'up':
            x_list[1], new_labels , new_train_idx, new_edge_indices, new_target_edge = oversample(features = x_list[1], labels = data['e'].y,  
            target_edge = target_edge, train_idx = train_idx, edge_indices = edge_indices, args= args, os_mode = 'up')
        elif os_mode == 'smote':
            x_list[1], new_labels , new_train_idx, new_edge_indices, new_target_edge  = oversample(features = x_list[1], labels = data['e'].y, 
            target_edge = target_edge, train_idx = train_idx, edge_indices = edge_indices, args= args, os_mode = 'smote')
        else:
            new_labels, new_train_idx, new_edge_indices, new_target_edge = data['e'].y, train_idx, edge_indices, target_edge
            
        x_list = encoder2(x_list, new_edge_indices)
        
        if os_mode == 'edge_sm':
                edge_em, new_labels, new_train_idx  = oversample(features =x_list[1], labels = data['e'].y, target_edge = target_edge, 
                train_idx = train_idx, edge_indices = None, args= args, os_mode = 'edge_sm')
        else: 
            edge_em = x_list[1][new_target_edge[:, 0]] * x_list[1][new_target_edge[:, 1]] 
        
        # Training    
        classifier.fit(edge_em[new_train_idx].cpu(), new_labels[new_train_idx].cpu())
        
        pred_prob = classifier.predict_proba(edge_em[new_train_idx].cpu()) 
        train_pred = (pred_prob[:, 1] >= thresh).astype(int)
        report = classification_report(new_labels[new_train_idx].cpu(), train_pred)
        roc_auc = roc_auc_score(new_labels[new_train_idx].cpu(), pred_prob[:, 1])
        print("Training report\n",report,"\nROC_AUC_SCORE:", roc_auc)
        
        labels = data['e'].y
        
        # Validation
        pred_prob = classifier.predict_proba(edge_em[val_idx].cpu()) 
        val_pred = (pred_prob[:, 1] >= thresh).astype(int)
        report = classification_report(labels[val_idx].cpu(), val_pred)
        roc_auc = roc_auc_score(labels[val_idx].cpu(), pred_prob[:, 1])
        print("\nValidation report",report,"\nROC_AUC_SCORE:", roc_auc)
        
        # Testing
        pred_prob = classifier.predict_proba(edge_em[test_idx].cpu()) 
        test_pred = (pred_prob[:, 1] >= thresh).astype(int)
        report = classification_report(labels[test_idx].cpu(), test_pred)
        roc_auc = roc_auc_score(labels[test_idx].cpu(), pred_prob[:, 1])
        print("\nTesting report\n",report,"\nROC_AUC_SCORE:", roc_auc)
        
        lines = report.split('\n')
        f1_0 = float(lines[2].split()[-2])  # F1 score for class 0
        f1_1 = float(lines[3].split()[-2])  # F1 score for class 1
        macro_f1 = float(lines[5].split()[-2])  # Macro F1 score
        acc = float(lines[6].split()[-2])  # Accuracy

        return f1_0, f1_1, macro_f1, acc, roc_auc

# Function to evalute all decoder related tasks
def het_decode(decoder_list, x_list, edge_indices, ar_edge_indices, args, train_mode, dataset):
    edge_ac = []
    loss_de = []
    edge_list = []
    acc = 0
 
    for i in range(len(decoder_list)):
        
        adj_new = decoder_list[i](x_list[1], x_list[i])
        ori_row_num = args.node_dim[1]
        ori_col_num = args.node_dim[i]
        
        adj_old = torch.zeros((ori_row_num, ori_col_num), dtype = torch.float32, device=x_list[0].device)
        adj_old[edge_indices[i][0], edge_indices[i][1]] = 1.0
        
        if ar_edge_indices is not None:
            adj_ar = torch.zeros((x_list[1].size(0), x_list[i].size(0)), dtype = torch.float32, device=x_list[0].device)
            adj_ar[ar_edge_indices[i][0], ar_edge_indices[i][1]] = 1.0
        else:
            adj_ar = None
  
        # edge_ac.append(F.l1_loss(adj_new[:ori_row_num, :ori_col_num], adj_old, reduction='mean'))
        loss_de.append(edge_loss(adj_new[:ori_row_num, :ori_col_num], adj_old))

        adj_new, acc = adj_gen(adj_new, adj_old, adj_ar, ori_row_num, ori_col_num, dataset)
        adj_new = adj_new.nonzero().t().contiguous()
        edge_ac.append(acc)
        
        if train_mode == 'preO' or train_mode == 'preo':
            edge_list.append(adj_new)
        else:
            edge_list.append(adj_new.detach())
            
    return edge_ac, loss_de, edge_list


def adj_gen(adj_new, adj_old, adj_ar, ori_row_num, ori_col_num, mode):
    threshold = 0.5
    adj_new = (adj_new >= threshold).float()
    
    if adj_ar is not None:
        adj_new = torch.mul(adj_ar, adj_new)

    diff = torch.abs(adj_new[:ori_row_num, :ori_col_num] - adj_old)
    correct_edges = (diff == 0).sum().item()
    total_edges = diff.numel()
    acc = correct_edges / total_edges
    
    if mode == 'Train': adj_new[:ori_row_num, :ori_col_num] = adj_old
    return adj_new, acc


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
9) edge_sm = smote on edges

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