import sys
sys.path.extend([ '../', '../../'])
import torch
import dataloader as dl
from args import Args
from model import Content_Agg, Neigh_Agg, EdgePredictor, Het_classify
from train import train_smote

# Set device to GPU if available, else use CPU
args = Args()
args.mp_imdb()
torch.cuda.empty_cache()

data = torch.load('data.pt')
file_path = 'output/up_ratio.txt'

device = args.device

# Send all x tensors to the device
data['mp1']['x'] = data['mp1']['x'].to(device)
data['mp2']['x'] = data['mp2']['x'].to(device)
data['mp3']['x'] = data['mp3']['x'].to(device)

# Send all y tensors to the device
data['mp']['y'] = data['mp']['y'].to(device)

# Send all edge_index tensors to the device
data['mp']['edge_index'] = data['mp']['edge_index'].to(device)
data['mp']['edge_weight'] = data['mp']['edge_weight'].to(device)


train_dict = {
    0: 'pret',
    1: 'preo', 
    2: 'preT',
    3: 'preO'
} 

up_ratios = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]

with open(file_path, "w") as file:   #, open(file_path2, "w") as file2:
    
    for num, up in enumerate(up_ratios):
        
        args.portion = up
        train_data_idx, val_data_idx, test_data_idx = [], [], []
        
        file.write(f'\nUp_ratio: {up}\n')
            
        for p in range(10):
            c_train_num = dl.train_num(data['mp'].y, args.im_class_num, args.class_samp_num[0], args.im_ratio)
            print(c_train_num, sum(c_train_num))
            train_idx, val_idx, test_idx, c_num_mat = dl.segregate(data['mp'].y, c_train_num, args.seed[p], args)
            
            train_data_idx.append(train_idx)
            val_data_idx.append(val_idx)
            test_data_idx.append(test_idx)
                        
        for k in range(0, 4):   
        
            train_mode = train_dict[k]
            os_mode = 'mp_smote'
            
            file.write(f'\nOs_mode: {os_mode}, train_mode: {train_mode}\n')     
            file.flush()
            Test_acc, Test_auc, Test_f1, auc_list, f1_list = [], [], [], [], []
    
            for p in range(10):
                    
                classifier = Het_classify(args.embed_dim, args.nclass, args.dropout)

                if k > 1:
                    encoder1 = torch.load('pretrained/encoder1.pth')
                    encoder2 = torch.load('pretrained/encoder2.pth')
                else:
                    encoder1 = Content_Agg(args.embed_dim, args.dropout)
                    encoder2 = Neigh_Agg(args.embed_dim, args.dropout)
                            
                if train_mode == 'preT' or train_mode == 'preO': decoder = torch.load('pretrained/decoder.pth')
                else: decoder = EdgePredictor(args.embed_dim)
            
                encoder1.to(device)
                encoder2.to(device)
                classifier.to(device)
                decoder.to(device)
                    
                train_idx, val_idx, test_idx = train_data_idx[p], val_data_idx[p], test_data_idx[p]
                
                test_acc_list, test_auc_list, test_f1_list, auc_cls_list, f1_cls_list = train_smote(data, encoder1, encoder2, classifier, 
                decoder, train_idx, val_idx, test_idx, args, train_mode = 'preO')
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
    
        
            