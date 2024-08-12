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
from smote import oversample
from train2 import train_smote
import cProfile

# Set device to GPU if available, else use CPU
args = Args()
args.aminer_edge_pred()
torch.cuda.empty_cache()

data = torch.load('edge_pred_data.pt')
file_path = 'output/result2.txt'
# file_path2 = 'output/cls_multi_scores.txt'

device = args.device
data = data.to(device)

# Send all x tensors to the device
data['e']['x'] = data['e']['x'].to(device)
data['p_title_embed']['x'] = data['p_title_embed']['x'].to(device)
data['p_abstract_embed']['x'] = data['p_abstract_embed']['x'].to(device)
data['p_net_embed']['x'] = data['p_net_embed']['x'].to(device)
data['p_a_net_embed']['x'] = data['p_a_net_embed']['x'].to(device)
# data['p_p_net_embed']['x'] = data['p_p_net_embed']['x'].to(device)
data['p_v_net_embed']['x'] = data['p_v_net_embed']['x'].to(device)
data['a_net_embed']['x'] = data['a_net_embed']['x'].to(device)
data['a_text_embed']['x'] = data['a_text_embed']['x'].to(device)
data['v_net_embed']['x'] = data['v_net_embed']['x'].to(device)
data['v_text_embed']['x'] = data['v_text_embed']['x'].to(device)

# Send all y tensors to the device
data['a']['y'] = data['a']['y'].to(device)
data['e']['y'] = data['e']['y'].to(device)

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

edge_indices = [ data['p', 'walk', 'a'].edge_index, data['p', 'walk', 'p'].edge_index, data['p', 'walk', 'v'].edge_index ]
target_edge = data['e']['x']


class_sample_nums = [500, 1000, 2000, 3000, 4000]
im_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

with open(file_path, "w") as file: #, open(file_path2, "w") as file2:

    for num in class_sample_nums:
        args.class_samp_num = [num, num, num+2000]
        file.write(f'\nSample Number: {num}\n')
        
        for k in args.thresh_dict:     

            os_mode = k
            train_mode = 'preo'
            file.write(f'\nOs_mode: {os_mode}\n')
            
            if k == 'no': im_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            else: im_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  

            for i in im_ratios:
                file.write(f'\nImb_ratio: {i}\n')
                args.im_ratio = [i]
                args.portion = 1/i
                           
                for run in range(5):
                    c_train_num = dl.train_num(data['e'].y, args.im_class_num, args.class_samp_num[0], args.im_ratio)
                    train_idx, val_idx, test_idx, c_num_mat = dl.segregate(data['e'].y, c_train_num, args)

                    encoder = torch.load('pretrained/encoder.pth')
                    encoder2 = torch.load('pretrained/encoder2.pth')
                    decoder_pa = torch.load('pretrained/decoder_pa.pth')
                    decoder_pp = torch.load('pretrained/decoder_pp.pth')
                    decoder_pv = torch.load('pretrained/decoder_pv.pth')
                    decoder_list = [decoder_pa, decoder_pp, decoder_pv]

                    encoder.to(device)
                    encoder2.to(device)
                    for decoder in decoder_list:
                        decoder.to(device)
                
                    torch.cuda.empty_cache()
                    f1_0, f1_1, macro_f1, acc, roc_auc = train_smote(data, edge_indices, target_edge, encoder, 
                    encoder2, decoder_list, train_idx, val_idx, test_idx, args, os_mode = os_mode, train_mode = train_mode)
                
                    file.write(f'f1_0: {f1_0}, f1_1: {f1_1}, macro_f1: {macro_f1}, acc: {acc}, roc_auc: {roc_auc}\n')
            
            