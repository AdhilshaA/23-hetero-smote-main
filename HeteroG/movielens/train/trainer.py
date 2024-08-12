import sys
sys.path.extend([ '../', '../../'])
import torch
import dataloader as dl
from args import Args
from model import Het_ConEn, Het_NetEn, EdgePredictor, Edge_classify
from train2 import train_smote

# Set device to GPU if available, else use CPU
args = Args()
args.movielens_edge_pred()
torch.cuda.empty_cache()

data = torch.load('ml_data.pt')
file_path = 'output/result3.txt'
# file_path2 = 'output/cls_multi_scores.txt'

device = args.device

# Send all x tensors to the device
data['u_embed']['x'] = data['u_embed']['x'].to(device)
data['u_net_embed']['x'] = data['u_net_embed']['x'].to(device)
data['m_embed']['x'] = data['m_embed']['x'].to(device)
data['m_net_embed']['x'] = data['m_net_embed']['x'].to(device)

# Send all y tensors to the device
data['u', 'edge', 'm']['y'] = data['u', 'edge', 'm']['y'].to(device)
data['u', 'edge', 'm']['edge_index'] = data['u', 'edge', 'm']['edge_index'].to(device)

# Send all edge_index tensors to the device
data['u', 'walk', 'u']['edge_index'] = data['u', 'walk', 'u']['edge_index'].to(device)
data['u', 'walk', 'm']['edge_index'] = data['u', 'walk', 'm']['edge_index'].to(device)
data['m', 'walk', 'u']['edge_index'] = data['m', 'walk', 'u']['edge_index'].to(device)
data['m', 'walk', 'm']['edge_index'] = data['m', 'walk', 'm']['edge_index'].to(device)


edge_indices_list = [[ data['u', 'walk', 'u'].edge_index, data['u', 'walk', 'm'].edge_index],
                [ data['m', 'walk', 'u'].edge_index, data['m', 'walk', 'm'].edge_index]]
target_edge = data['u', 'edge', 'm']['edge_index'].transpose(0, 1)


train_dict = {
    0: 'no',
    1: 'up',
    2: 'smote',
    3: 'reweight',
    4: 'embed_sm',
    5: 'em_smote',
    6: 'edge_sm',
    7: 'pret',
    8: 'preo',
    9: 'preT', 
    10: 'preO',
    11: 'noFT',
    12: 'preT',
    13: 'preO'
} # 8, 9: pre enc + pre dec; 11,12: only pre dec

# # up_ratios = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
# class_sample_nums = [50, 80, 100, 150]
im_ratios = [0.1, 0.2, 0.4, 0.6]
r=0
z=14

with open(file_path, "w") as file:  
    
    for num, i in enumerate(im_ratios):
        
        if num == 0: r = 4
        else: r = 0
        
        args.im_ratio = [0.1, i, 0.5, 0.6]
        train_data_idx, val_data_idx, test_data_idx = [], [], []
        
        file.write(f'\nIm_ratio: {args.im_ratio}\n')
        
        for p in range(3):
            c_train_num = dl.train_num(data['u', 'edge', 'm']['y'], args.im_class_num, args.class_samp_num[0], args.im_ratio)
            print(c_train_num, sum(c_train_num))
            train_idx, val_idx, test_idx, c_num_mat = dl.segregate(data['u', 'edge', 'm']['y'], c_train_num, args.seed[p], args)
            
            train_data_idx.append(train_idx)
            val_data_idx.append(val_idx)
            test_data_idx.append(test_idx)
            
        for k in range(r, z):   
        
            if k < 7:
                train_mode = ''
                os_mode = train_dict[k]
            else:
                train_mode = train_dict[k]
                os_mode = 'gsm'
            
            if k > 11: file.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}2\n')  
            else: file.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}\n')     
            file.flush()
            Test_acc, Test_auc, Test_f1, auc_list, f1_list = [], [], [], [], []
            
            for p in range(3):
                
                classifier = Edge_classify(args.embed_dim, args.nclass, args.dropout)

                if k == 9 or k == 10:
                    encoder1 = torch.load('pretrained/encoder1.pth')
                    encoder2 = torch.load('pretrained/encoder2.pth')
                else:
                    encoder1 = Het_ConEn(args.embed_dim, args, args.dropout)
                    encoder2 = Het_NetEn(args.embed_dim, args.dropout)
                    
                if train_dict[k] == 'preT' or train_dict[k] == 'preO' or train_dict[k] == 'noFT':
                    decoder_uu = torch.load('pretrained/decoder_uu.pth')
                    decoder_um = torch.load('pretrained/decoder_um.pth')
                    decoder_mu = torch.load('pretrained/decoder_mu.pth')
                    decoder_mm = torch.load('pretrained/decoder_mm.pth')
                else: 
                    decoder_uu = EdgePredictor(args.embed_dim)
                    decoder_um = EdgePredictor(args.embed_dim)
                    decoder_mu = EdgePredictor(args.embed_dim)
                    decoder_mm = EdgePredictor(args.embed_dim)
            
                decoder_list1 = [decoder_uu, decoder_um]
                decoder_list2 = [decoder_mu, decoder_mm]

                encoder1.to(device)
                encoder2.to(device)
                classifier.to(device)
                for decoder in decoder_list1: decoder.to(device)
                for decoder in decoder_list2: decoder.to(device)
                    
                train_idx, val_idx, test_idx = train_data_idx[p], val_data_idx[p], test_data_idx[p]
            
                test_acc_list, test_auc_list, test_f1_list, auc_cls_list, f1_cls_list= train_smote(data,edge_indices_list,target_edge, encoder1, 
                encoder2, classifier, decoder_list1, decoder_list2, train_idx, val_idx, test_idx, args, os_mode=os_mode, train_mode=train_mode)
                Test_acc.append(test_acc_list)
                Test_auc.append(test_auc_list)
                Test_f1.append(test_f1_list)
                auc_list.append(auc_cls_list)
                f1_list.append(f1_cls_list)
                torch.cuda.empty_cache()
        
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
            
            