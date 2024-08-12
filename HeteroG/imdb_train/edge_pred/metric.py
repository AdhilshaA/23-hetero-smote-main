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

threshold = 0.334

def accuracy(outputs, labels):
    # print(outputs)
    # print(labels)
    # threshold = 0.27
    predicted = (outputs > threshold).type_as(labels)      # Returns a tuple of two numbers, 1st: Largest element in "1" dimension. 
    correct = (predicted == labels).sum().item()           # 2nd: The index of 1st in the "1" dimension. So we extract [1] dimension of the tuple
    total = labels.size(0)
    acc = correct / total
    return acc


def evaluate_class_performance(outputs, labels, args):
    num_classes = outputs.shape[1]
    auc_roc_scores = []
    f1_scores = []
    # bin_output = torch.softmax(outputs, 1).to(outputs.device)
    # print(bin_output)
    # threshold = 0.27
    # best_threshold = args.best_threshold

 
    for class_idx in range(num_classes):
        y_pred_prob_class = (outputs > threshold).flatten().to(int).cpu().numpy()
        y_true_class = labels.flatten().to(int).cpu().numpy()
        print(outputs, y_pred_prob_class, y_true_class)
        # y_true_class = (labels == class_idx).to(int).cpu().numpy()
        # y_pred_prob_class = bin_output[:, class_idx].cpu().numpy()
       
    
        # Find the optimal threshold that maximizes F1 score
        # thresholds = np.linspace(0.1, 0.9, 50)  # Adjust the range and number of thresholds
        # f1_scores_class = [f1_score(y_true_class, (y_pred_prob_class >= th).astype(int)) for th in thresholds]
        # best_threshold = thresholds[np.argmax(f1_scores_class)]
        
        auc_roc = roc_auc_score(y_true_class, y_pred_prob_class)
        f1 = f1_score(y_true_class, y_pred_prob_class)
        # print(f"class {class_idx}: {best_threshold}, ", end ='')
        
        auc_roc_scores.append(auc_roc)
        f1_scores.append(f1)

        print(f"Class {class_idx}:", f"AUC-ROC- {auc_roc:.4f},", f"F1 Score- {f1:.4f}; ", end="")

    macro_avg_auc_roc = np.mean(auc_roc_scores)
    macro_avg_f1 = np.mean(f1_scores)

    print(f"Macro-Average AUC-ROC: {macro_avg_auc_roc:.4f};", f"Macro-Average F1 Score: {macro_avg_f1:.4f}")
    return macro_avg_auc_roc, macro_avg_f1, auc_roc_scores, f1_scores