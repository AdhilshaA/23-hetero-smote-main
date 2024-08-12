import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.extend([ '../', '../../'])
import torch
import torch.nn as nn
import torch.optim as optim
from Data import input_data
import dataloader as dl
from args import Args
from model import Het_En, Classifier, EdgePredictor, Het_classify
from smote import oversample
from train import train_smote, test_smote

# Set device to GPU if available, else use CPU
args = Args()
args.imdb()
torch.cuda.empty_cache()

data = torch.load('../data/data.pt')
file_path = 'output/cls_multi_main.txt'
file_path2 = 'output/cls_multi_scores.txt'

device = args.device
data = data.to(device)

# Send all x tensors to the device
data['m_text_embed']['x'] = data['m_text_embed']['x'].to(device)
data['m_net_embed']['x'] = data['m_net_embed']['x'].to(device)
data['m_a_net_embed']['x'] = data['m_a_net_embed']['x'].to(device)
data['m_d_net_embed']['x'] = data['m_d_net_embed']['x'].to(device)
data['a_net_embed']['x'] = data['a_net_embed']['x'].to(device)
data['a_text_embed']['x'] = data['a_text_embed']['x'].to(device)
data['d_net_embed']['x'] = data['d_net_embed']['x'].to(device)
data['d_text_embed']['x'] = data['d_text_embed']['x'].to(device)

# Send all y tensors to the device
data['m']['y'] = data['m']['y'].to(device)

# Send all edge_index tensors to the device
data['m', 'walk', 'm']['edge_index'] = data['m', 'walk', 'm']['edge_index'].to(device)
data['m', 'walk', 'a']['edge_index'] = data['m', 'walk', 'a']['edge_index'].to(device)
data['m', 'walk', 'd']['edge_index'] = data['m', 'walk', 'd']['edge_index'].to(device)
data['a', 'walk', 'm']['edge_index'] = data['a', 'walk', 'm']['edge_index'].to(device)
data['a', 'walk', 'a']['edge_index'] = data['a', 'walk', 'a']['edge_index'].to(device)
data['a', 'walk', 'd']['edge_index'] = data['a', 'walk', 'd']['edge_index'].to(device)
data['d', 'walk', 'm']['edge_index'] = data['d', 'walk', 'm']['edge_index'].to(device)
data['d', 'walk', 'a']['edge_index'] = data['d', 'walk', 'a']['edge_index'].to(device)
data['d', 'walk', 'd']['edge_index'] = data['d', 'walk', 'd']['edge_index'].to(device)

edge_indices = [ data['m', 'walk', 'm'].edge_index, data['m', 'walk', 'a'].edge_index, data['m', 'walk', 'd'].edge_index ]


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
# class_sample_nums = [20, 40, 60, 80]

with open(file_path, "w") as file, open(file_path2, "w") as file2:
    
    for k in range(0, 10):   
        
        if k < 5:
            train_mode = ''
            os_mode = train_dict[k]
        else:
            train_mode = train_dict[k]
            os_mode = 'gsm'
        
        args.best_threshold = args.thresh_dict[k]
        file.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}\n')
        
        # if k == 0: im_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # else: im_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  

        # for i in class_sample_nums:
        # for i in im_ratios:
        # for i in up_ratios:
            
        Test_acc = []
        Test_auc = []
        Test_f1 = []
        auc_list = []
        f1_list = []
        
        # args.im_ratio = [i]
        # args.portion = i
        # args.class_samp_num = [i, i+40, 2*i+40]
        
        for j in range(10):
            
            c_train_num = dl.train_num(data['m'].y, args.im_class_num, args.class_samp_num[0], args.im_ratio)
            print(c_train_num, sum(c_train_num))
            train_idx, val_idx, test_idx, c_num_mat = dl.segregate(data['m'].y, c_train_num, args)

            encoder = Het_En(args.embed_dim, args.dropout)
            classifier = Het_classify(args.embed_dim, args.nclass, args.dropout)
        
            if train_dict[k] == 'preT' or train_dict[k] == 'preO':
                decoder_m = torch.load('pre_decoder/decoder_m.pth')
                decoder_a = torch.load('pre_decoder/decoder_a.pth')
                decoder_d = torch.load('pre_decoder/decoder_d.pth')
            else:
                decoder_m = EdgePredictor(args.embed_dim)
                decoder_a = EdgePredictor(args.embed_dim)
                decoder_d = EdgePredictor(args.embed_dim)
        
            decoder_list = [decoder_m, decoder_a, decoder_d]
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
        

        # file.write(f'\nClass Sample Num: {args.class_samp_num}\nTest_acc:\n')
        file.write(f'\nTest_acc:\n')
        # file.write(f'\nUp_ratio: {args.portion}\nTest_acc:\n')
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
        
            