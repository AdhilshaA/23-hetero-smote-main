import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def edge_loss(adj_old, adj_new):

    edge_num = adj_new.nonzero().shape[0]
    total_num = adj_new.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_old.new(adj_new.shape).fill_(1.0)
    weight_matrix[adj_new <= 0.5] = neg_weight

    loss = torch.sum(weight_matrix * (adj_old - adj_new) ** 2)/total_num

    return loss

def accuracy(outputs, labels):
    predicted = torch.max(outputs, 1)[1].type_as(labels)   # Returns a tuple of two numbers, 1st: Largest element in "1" dimension. 
    correct = (predicted == labels).sum().item()           # 2nd: The index of 1st in the "1" dimension. So we extract [1] dimension of the tuple
    total = labels.size(0)
    acc = correct / total
    return acc


def evaluate_class_performance(outputs, labels, thresh_list, dataset, args):
    num_classes = outputs.shape[1]
    auc_roc_scores = []
    f1_scores = []
    # print(outputs)
    bin_output = torch.softmax(outputs, 1).to(outputs.device)
    # print(bin_output)
    # print(bin_output)
    thresh_list = []
    best_threshold = args.best_threshold

 
    for class_idx in range(num_classes):
        
        y_true_class = (labels == class_idx).to(int).cpu().numpy()
        y_pred_prob_class = bin_output[:, class_idx].cpu().numpy()
       
    
        # Find the optimal threshold that maximizes F1 score
        if dataset != 'Test':
            thresholds = np.linspace(0.1, 0.9, 50)  # Adjust the range and number of thresholds
            f1_scores_class = [f1_score(y_true_class, (y_pred_prob_class >= th).astype(int)) for th in thresholds]
            best_threshold = thresholds[np.argmax(f1_scores_class)]
        
        auc_roc = roc_auc_score(y_true_class, y_pred_prob_class)
        
        if dataset != 'Test':
            f1 = f1_score(y_true_class, (y_pred_prob_class >= best_threshold).astype(int))
            thresh_list.append(best_threshold)
            print(f"class {class_idx}: {best_threshold}, ", end ='')
        
        else: 
            f1 = f1_score(y_true_class, (y_pred_prob_class >= best_threshold[class_idx]).astype(int))
            print(f"class {class_idx}: {best_threshold[class_idx]}, ", end ='')
            
        
        auc_roc_scores.append(auc_roc)
        f1_scores.append(f1)
        print(f"Class {class_idx}:", f"AUC-ROC- {auc_roc:.4f},", f"F1 Score- {f1:.4f}; ", end="")

    macro_avg_auc_roc = np.mean(auc_roc_scores)
    macro_avg_f1 = np.mean(f1_scores)
    args.best_threshold = thresh_list

    print(f"Macro-Average AUC-ROC: {macro_avg_auc_roc:.4f};", f"Macro-Average F1 Score: {macro_avg_f1:.4f}")
    return macro_avg_auc_roc, macro_avg_f1, auc_roc_scores, f1_scores




'''
no: [0.23, 0.21, 0.32, 0.30]
up: [0.29, 0.23, 0.28, 0.31]
smote: [0.27, 0.26, 0.26, 0.31]
reweight: [0.30, 0.25, 0.31, 0.31]
embed_sm: [0.26, 0.28, 0.31, 0.28]
no_rec_preT: [0.30, 0.26, 0.26, 0.33]
no_rec preO: [0.29, 0.23, 0.29, 0.33]
rec_preT: [0.26, 0.23, 0.29, 0.29]
rec_prO: [0.33, 0.25, 0.26, 0.31]
newG: [0.31, 0.28, 0.29, 0.26]
noFT: [0.29, 0.26, 0.31, 0.26]
'''
