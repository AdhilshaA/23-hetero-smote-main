import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def edge_loss(adj_old, adj_new):

    edge_num = adj_new.nonzero().shape[0]
    total_num = adj_new.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_old.new(adj_new.shape).fill_(1.0)
    weight_matrix[adj_new <= 0.5] = neg_weight

    loss = torch.sum(weight_matrix * (adj_old - adj_new) ** 2)

    return loss

def accuracy(outputs, labels):
    predicted = torch.max(outputs, 1)[1].type_as(labels)   # Returns a tuple of two numbers, 1st: Largest element in "1" dimension. 
    correct = (predicted == labels).sum().item()           # 2nd: The index of 1st in the "1" dimension. So we extract [1] dimension of the tuple
    total = labels.size(0)
    acc = correct / total
    return acc


def evaluate_class_performance(outputs, labels):
    num_classes = outputs.shape[1]
    auc_roc_scores = []
    f1_scores = []
    bin_output = torch.softmax(outputs, 1)

    for class_idx in range(num_classes):
        
        y_true_class = (labels == class_idx).to(int)
        y_pred_prob_class = bin_output[:, class_idx].cpu().detach().numpy()
        
        auc_roc = roc_auc_score(y_true_class, y_pred_prob_class)
        f1 = f1_score(y_true_class, (y_pred_prob_class >= 0.5).astype(int))
        
        auc_roc_scores.append(auc_roc)
        f1_scores.append(f1)

        print(f"Class {class_idx}:" f"AUC-ROC- {auc_roc:.4f},", f"F1 Score- {f1:.4f}; ", end="")

    macro_avg_auc_roc = np.mean(auc_roc_scores)
    macro_avg_f1 = np.mean(f1_scores)

    print(f"Macro-Average AUC-ROC: {macro_avg_auc_roc:.4f}," f"Macro-Average F1 Score: {macro_avg_f1:.4f}")
