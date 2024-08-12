import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.extend([ '../', '../../'])
import torch
import torch.nn as nn
import torch.optim as optim
from data import input_data
import dataloader as dl
from args import Args
from model import Het_En, Classifier, EdgePredictor, Het_classify
from smote import oversample
from train import train_smote, test_smote

# Set device to GPU if available, else use CPU
args = Args()
args.aminer_train()
torch.cuda.empty_cache()

data = torch.load('../../data/aminer/am_data.pt')
# print(data)
file_path = 'output_imb2.txt'
file_path2 = 'output_cls.txt'

# print(data['a','walk','p'].edge_index)
device = args.device
data = data.to(device)

# Send all x tensors to the device
data['p_title_embed']['x'] = data['p_title_embed']['x'].to(device)
data['p_abstract_embed']['x'] = data['p_abstract_embed']['x'].to(device)
data['p_net_embed']['x'] = data['p_net_embed']['x'].to(device)
data['p_a_net_embed']['x'] = data['p_a_net_embed']['x'].to(device)
data['p_p_net_embed']['x'] = data['p_p_net_embed']['x'].to(device)
data['p_v_net_embed']['x'] = data['p_v_net_embed']['x'].to(device)
data['a_net_embed']['x'] = data['a_net_embed']['x'].to(device)
data['a_text_embed']['x'] = data['a_text_embed']['x'].to(device)
data['v_net_embed']['x'] = data['v_net_embed']['x'].to(device)
data['v_text_embed']['x'] = data['v_text_embed']['x'].to(device)

# Send all y tensors to the device
data['a']['y'] = data['a']['y'].to(device)

# Send all edge_index tensors to the device
data['a', 'walk', 'a']['edge_index'] = data['a', 'walk', 'a']['edge_index'].to(device)
data['a', 'walk', 'p']['edge_index'] = data['a', 'walk', 'p']['edge_index'].to(device)
data['a', 'walk', 'v']['edge_index'] = data['a', 'walk', 'v']['edge_index'].to(device)
data['p', 'walk', 'a']['edge_index'] = data['p', 'walk', 'a']['edge_index'].to(device)
data['p', 'walk', 'p']['edge_index'] = data['p', 'walk', 'p']['edge_index'].to(device)
data['p', 'walk', 'v']['edge_index'] = data['p', 'walk', 'v']['edge_index'].to(device)
data['v', 'walk', 'a']['edge_index'] = data['v', 'walk', 'a']['edge_index'].to(device)
data['v', 'walk', 'p']['edge_index'] = data['v', 'walk', 'p']['edge_index'].to(device)
data['v', 'walk', 'v']['edge_index'] = data['v', 'walk', 'v']['edge_index'].to(device)

edge_indices = [ data['a', 'walk', 'a'].edge_index, data['a', 'walk', 'p'].edge_index, data['a', 'walk', 'v'].edge_index ]



    
thresh_dict = {
    0: [0.23, 0.21, 0.32, 0.30],
    1: [0.29, 0.23, 0.28, 0.31],
    2: [0.27, 0.26, 0.26, 0.31],
    3: [0.30, 0.25, 0.31, 0.31],
    4: [0.26, 0.28, 0.31, 0.28],
    5: [0.30, 0.26, 0.26, 0.33],
    6: [0.29, 0.23, 0.29, 0.33],
    7: [0.26, 0.23, 0.29, 0.29],
    8: [0.33, 0.25, 0.26, 0.31],
    9: [0.29, 0.26, 0.31, 0.26],
    10: [0.31, 0.28, 0.29, 0.26]
}

train_dict = {
    0: 'no',
    1: 'up',
    2: 'smote',
    3: 'reweight',
    4: 'embed_sm',
    5: 'pret',
    6: 'preo',
    7: 'preT',
    8: 'preO',
    9: 'noFT',
    10: 'newG'
}

# up_ratios = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
# class_sample_nums = [20, 30, 40]

with open(file_path, "w") as file, open(file_path2, "w") as file2:
    
    for k in range(0, 10):   
        
        if k < 5:
            train_mode = ''
            os_mode = train_dict[k]
        else:
            train_mode = train_dict[k]
            os_mode = 'gsm'
        
        args.best_threshold = thresh_dict[k]
        file.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}\n')
        
        # if k == 0: im_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # else: im_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  

        # for i in class_sample_nums:
            
        Test_acc = []
        Test_auc = []
        Test_f1 = []
        auc_list = []
        f1_list = []
        
        
        for j in range(10):
            
            c_train_num = dl.train_num(data['a'].y, args.im_class_num, args.class_sample_num, args.im_ratio)
            print(c_train_num, sum(c_train_num))
            train_idx, val_idx, test_idx, c_num_mat = dl.segregate(data['a'].y, c_train_num)

            encoder = Het_En(args.embed_dim, args.dropout)
            classifier = Het_classify(args.embed_dim, args.nclass, args.dropout)
        
            if train_dict[k] == 'preT' or train_dict[k] == 'preO':
                decoder_a = torch.load('pre_decoder/decoder_a.pth')
                decoder_p = torch.load('pre_decoder/decoder_p.pth')
                decoder_v = torch.load('pre_decoder/decoder_v.pth')
            else:
                decoder_a = EdgePredictor(args.embed_dim)
                decoder_p = EdgePredictor(args.embed_dim)
                decoder_v = EdgePredictor(args.embed_dim)
        
            decoder_list = [decoder_a, decoder_p, decoder_v]
            #print(features.shape)
            encoder.to(device)
            classifier.to(device)
            for decoder in decoder_list:
                decoder.to(device)
        
            test_acc_list, test_auc_list, test_f1_list, auc_cls_list, f1_cls_list = train_smote(data, edge_indices, encoder, classifier, decoder_list, train_idx, val_idx, test_idx, args, os_mode = os_mode, train_mode = train_mode)
            Test_acc.append(test_acc_list)
            Test_auc.append(test_auc_list)
            Test_f1.append(test_f1_list)
            auc_list.append(auc_cls_list)
            f1_list.append(f1_cls_list)
            torch.cuda.empty_cache()
        

        # file.write(f'\nClass Sample Num: {args.class_sample_num}\nTest_acc:\n')
        file.write(f'\nTest_acc:\n')
        for row in Test_acc:
            row_str = " ".join(map(str, row))
            file.write(row_str + "\n")
        file.write(f'\nTest_auc:\n')    
        for row in Test_auc:
            row_str = " ".join(map(str, row))
            file.write(row_str + "\n")
        file.write(f'\nTest_f1:\n')  
        for row in Test_f1:
            row_str = " ".join(map(str, row))
            file.write(row_str + "\n")
            

        file2.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}\n\nClass AUC score:\n')
        for row in auc_list:
            row_str = " ".join(map(str, row))
            file2.write(row_str + "\n")
        file2.write(f'\nClass F1 Score:\n')  
        for row in f1_list:  
            row_str = " ".join(map(str, row))
            file2.write(row_str + "\n")
        
            