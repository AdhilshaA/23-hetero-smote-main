import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.extend([ '../', '../../'])
import torch
from Data import input_data
import dataloader as dl
from args import Args
from model import Het_ConEn, Het_NetEn, EdgePredictor, Het_classify
from smote import oversample
from train import train_smote, test_smote

# Set device to GPU if available, else use CPU
args = Args()
args.yelp()
torch.cuda.empty_cache()

data = torch.load('../data2.pt')
file_path = 'output/up_scale.txt'
# file_path2 = 'output/cls_multi_scores.txt'

device = args.device

# Send all x tensors to the device
data['r_embed']['x'] = data['r_embed']['x'].to(device)
data['u_embed']['x'] = data['u_embed']['x'].to(device)
data['b_embed']['x'] = data['b_embed']['x'].to(device)

# Send all y tensors to the device
data['b']['y'] = data['b']['y'].to(device)

# Send all edge_index tensors to the device
data['r', 'walk', 'r']['edge_index'] = data['r', 'walk', 'r']['edge_index'].to(device)
data['r', 'walk', 'u']['edge_index'] = data['r', 'walk', 'u']['edge_index'].to(device)
data['r', 'walk', 'b']['edge_index'] = data['r', 'walk', 'b']['edge_index'].to(device)
data['u', 'walk', 'r']['edge_index'] = data['u', 'walk', 'r']['edge_index'].to(device)
data['u', 'walk', 'u']['edge_index'] = data['u', 'walk', 'u']['edge_index'].to(device)
data['u', 'walk', 'b']['edge_index'] = data['u', 'walk', 'b']['edge_index'].to(device)
data['b', 'walk', 'r']['edge_index'] = data['b', 'walk', 'r']['edge_index'].to(device)
data['b', 'walk', 'u']['edge_index'] = data['b', 'walk', 'u']['edge_index'].to(device)
data['b', 'walk', 'b']['edge_index'] = data['b', 'walk', 'b']['edge_index'].to(device)

edge_indices = [ data['b', 'walk', 'b'].edge_index, data['b', 'walk', 'u'].edge_index, data['b', 'walk', 'r'].edge_index ]


train_dict = {
    0: 'no',
    1: 'up',
    2: 'smote',
    3: 'reweight',
    4: 'embed_sm',
    5: 'em_smote',
    6: 'pret',
    7: 'preo',
    8: 'preT',
    9: 'preO',
    10: 'noFT',
}

up_ratios = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
class_sample_nums = [500] #[500, 200, 100, 50]
args.class_samp_num = [500, 600, 700]

with open(file_path, "w") as file:   #, open(file_path2, "w") as file2:
    
     for u, up in enumerate(up_ratios):
        
        args.portion = up
        file.write(f'\nUp_ratio:{up}\n')
        
        if up == 1.0: im_ratios = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8]
        else: im_ratios = [0.1]
        
        for im, i in enumerate(im_ratios):
    
            args.im_ratio = [i]
            train_data_idx, val_data_idx, test_data_idx = [], [], []
            
            file.write(f'\nIm_ratio: {i}\n')
            
            for p in range(5):
                c_train_num = dl.train_num(data['b'].y, args.im_class_num, args.class_samp_num[0], args.im_ratio)
                print(c_train_num, sum(c_train_num))
                train_idx, val_idx, test_idx, c_num_mat = dl.segregate(data['b'].y, c_train_num, args.seed[p], args)
                
                train_data_idx.append(train_idx)
                val_data_idx.append(val_idx)
                test_data_idx.append(test_idx)
             
            for k in range(0, 10):   
                
                if k < 6:
                    train_mode = ''
                    os_mode = train_dict[k]
                else:
                    train_mode = train_dict[k]
                    os_mode = 'gsm'
                
                file.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}\n')    
                file.flush()               
                Test_acc, Test_auc, Test_f1, auc_list, f1_list = [], [], [], [], []
                
                for p in range(5):
                    
                    classifier = Het_classify(args.embed_dim, args.nclass, args.dropout)
                
                    if train_dict[k] == 'preT' or train_dict[k] == 'preO':
                        encoder1 = torch.load('pretrained/encoder1.pth')
                        encoder2 = torch.load('pretrained/encoder2.pth')
                        decoder_b = torch.load('pretrained/decoder_b.pth')
                        decoder_u = torch.load('pretrained/decoder_u.pth')
                        decoder_r = torch.load('pretrained/decoder_r.pth')
                    else: 
                        encoder1 = Het_ConEn(args.embed_dim, args.dropout)
                        encoder2 = Het_NetEn(args.embed_dim, args.dropout)
                        decoder_b = EdgePredictor(args.embed_dim)
                        decoder_u = EdgePredictor(args.embed_dim)
                        decoder_r = EdgePredictor(args.embed_dim)
                
                    decoder_list = [decoder_b, decoder_u, decoder_r]
                    #print(features.shape)
                    encoder1.to(device)
                    encoder2.to(device)
                    classifier.to(device)
                    for decoder in decoder_list:
                        decoder.to(device)
                        
                    train_idx, val_idx, test_idx = train_data_idx[p], val_data_idx[p], test_data_idx[p]
                
                    test_acc_list, test_auc_list, test_f1_list, auc_cls_list, f1_cls_list = train_smote(data, edge_indices, encoder1, 
                    encoder2, classifier, decoder_list, train_idx, val_idx, test_idx, args, os_mode = os_mode, train_mode = train_mode)
                    Test_acc.append(test_acc_list)
                    Test_auc.append(test_auc_list)
                    Test_f1.append(test_f1_list)
                    auc_list.append(auc_cls_list)
                    f1_list.append(f1_cls_list)
                    torch.cuda.empty_cache()
            

                # file.write(f'\nClass Sample Num: {args.class_samp_num}\nTest_acc:\n')
                # file.write(f'\nUp_ratio: {args.portion}\nTest_acc:\n')
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
                file.flush()
                
                file.write(f'\nClass AUC Score:\n')  
                for row in auc_list:
                    row_str = " ".join(map(str, row))
                    file.write(row_str + "\n")
                file.write(f'\nClass F1 Score:\n')  
                for row in f1_list:  
                    row_str = " ".join(map(str, row))
                    file.write(row_str + "\n")
                file.flush()

            # file2.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}\n\nClass AUC score:\n')
            # for row in auc_list:
            #     row_str = " ".join(map(str, row))
            #     file2.write(row_str + "\n")
            # file2.write(f'\nClass F1 Score:\n')  
            # for row in f1_list:  
            #     row_str = " ".join(map(str, row))
            #     file2.write(row_str + "\n")
        
            