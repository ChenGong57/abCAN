import os
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from modeling import Decoder
from process_data import load_file, DataLoader, Completize, data_enhance

def split_dataset(data_path, k_fold = 0):
    '''分割原始数据为子dataset'''

    # 提取所有数据的index
    items = load_file(data_path)
    num_list = []
    for item in items: 
        num = item['No'].split('_')[-1]
        if num in num_list: continue
        num_list.append(num)
    random.shuffle(num_list)

    # 对index进行随机划分
    if k_fold == 0:      # 对于普通4：1的情况
        train_index, valid_index = [], []
        for i, index in enumerate(num_list):
            if i <= len(num_list) * 4 // 5: train_index.append(index)
            else: valid_index.append(index)
        split_index = [train_index, valid_index]
    else:
        split_index = []
        for i in range(k_fold):
            if i == 0: start = 0
            else: start = len(num_list) * i // k_fold + 1
            end = len(num_list) * (i+1) // k_fold + 1
            fold_list = num_list[start:end]
            split_index.append(fold_list)

    # 返回数据列表
    items_list = []
    for _ in split_index: items_list.append([])
    for item in items:
        num = item['No'].split('_')[-1]
        for i, split_list in enumerate(split_index):
            if num in split_list:
                items_list[i].append(item)
                continue

    return items_list
        

def form_loader(dataset_list, k_fold = 0, valid_index = 0):

    '''从dataset生成可用于训练的train/valid loader'''

    if k_fold == 0:
        loader_list = [] 
        for i,dataset in enumerate(dataset_list):
            if i == 0: 
                dataset = data_enhance(dataset)      # 训练集数据增强
            loader = DataLoader(dataset, args.batch_size, args.batch_tokens)
            loader_list.append(loader)
        train_loader, valid_loader = loader_list
    else:
        train_dataset = []
        for i,dataset in enumerate(dataset_list):
            if i != valid_index: 
                train_dataset.extend(dataset)
            else: 
                valid_loader = DataLoader(dataset, args.batch_size, args.batch_tokens)
        train_dataset = data_enhance(train_dataset)
        train_loader = DataLoader(train_dataset, args.batch_size, args.batch_tokens)
    
    return train_loader, valid_loader


def evaluate(model, loader):
    model.eval()
    val_rmsd = []

    # 在评估模式下不进行梯度计算
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            proteins = Completize(batch, args.esm)
            X, oS, mS, L, mask, ddG_true = proteins.hbatch
            ddG_pre = model(X, oS, mS, L, mask)
            print('ddG_pre:', ddG_pre)
            if torch.isnan(ddG_pre).any():
                raise ValueError("Predictions contain NaNs, terminating execution.")
            rmsd = torch.sqrt(torch.sum((ddG_true - ddG_pre)**2)/ddG_pre.shape[0])
            val_rmsd.append(rmsd.item())

            if i == 0:
                predictions = ddG_pre
                trues = ddG_true
            else:
                predictions = torch.cat([predictions, ddG_pre], dim=-1)
                trues = torch.cat([trues, ddG_true], dim=-1)

            # break
            
        try: 
            pc, p_value = pearsonr(predictions.cpu(), trues.cpu())
            return sum(val_rmsd) / len(val_rmsd), pc, p_value, trues.cpu(), predictions.cpu()
        except: 
            print('predictions:', predictions)
            print('trues:', trues)
            return sum(val_rmsd) / len(val_rmsd), trues.cpu(), predictions.cpu()



def plt_prediction(ddG_true, ddG_pre, title, rmsd, pc):

    # 绘制预测结果散点图
    sorted_indices = np.argsort(ddG_true)
    ddG_true = ddG_true[sorted_indices]
    ddG_pre = ddG_pre[sorted_indices]
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(ddG_true)), ddG_true, color='blue', label='True ddG')
    plt.scatter(range(len(ddG_pre)), ddG_pre, color='red', label='Predicted ddG', s=5)
    plt.xlabel('Antibody-antigein complex')
    plt.ylabel('ddG Values')
    plt.title(f'{title} set prediction results')
    plt.legend(loc='upper left')
    plt.text(0.8,0.05,f'RMSD: {rmsd:.2f}\nPearson: {pc:.2f}',
            horizontalalignment='left', verticalalignment='baseline', transform=plt.gca().transAxes)
    metric_path = os.path.join(cluster_path,title+'_pre_curve.png')
    plt.savefig(metric_path)


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/home/gongchen/model/data/pair_temp.json')
parser.add_argument('--test_path', default='/home/gongchen/model/data/SKEMPI1.1/pair_temp.json')
parser.add_argument('--save_dir', default='/home/gongchen/model/ckpts/final')
parser.add_argument('--load_model', default='/home/gongchen/model/ckpts/best/check/model603_652.ckpt')

parser.add_argument('--batch_tokens', type=int, default=70)
parser.add_argument('--L_max', type=int, default=450)
parser.add_argument('--k_fold', type=int, default=0)
parser.add_argument('--esm', default=None)      # None
parser.add_argument('--depth', type=int, default=6)
parser.add_argument('--vocab_size', type=int, default=21)   # 氨基酸index
parser.add_argument('--block_size', type=int, default=16)    # 粗粒化的每一个模块大小,8
parser.add_argument('--num_rbf', type=int, default=16)
parser.add_argument('--pos_dims', type=int, default=64)

parser.add_argument('--k_neighbors', type=int, default=9)
parser.add_argument('--seed', type=int, default=571) 
parser.add_argument('--epochs', type=int, default=500)   # 10

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--hidden_size', type=int, default=128)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

os.makedirs(args.save_dir, exist_ok=True)


# 预测
model = Decoder(args).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
model_ckpt, opt_ckpt, sche_ckpt, model_args = torch.load(args.load_model)
model.load_state_dict(model_ckpt)
optimizer.load_state_dict(opt_ckpt)
scheduler.load_state_dict(sche_ckpt)

# 加载数据集
bench_items = load_file(args.test_path)

combined_path = '/home/gongchen/model/GeoPPI-master/compare/combined_temp.json'
combined_items = load_file(combined_path)
cm_list = ['LI', 'IL', 'LV', 'VL',  'DE', 'ED', 'ST', 'TS', 'FY', 'YF', 'FW', 'WF', 'YW', 'WY', 'KR', 'RK', 'KH', 'HK', 'RH', 'HR']
cm_pdb = ['1dvf', '1vfb', '1kiq', '2jel', '1nca', '1jrh', '1nmb', '3hfm', '2vlj']
check_pdb = ['1JTG', '1KTZ', '1FFW', '1AK4']
# for n, complex in enumerate(cm_pdb):
complex_list = []
for item in combined_items:
    if item['pdb'] in complex_list: continue
    complex_list.append(item['pdb'])

index_list = [1,2,3,4,5,6,7,8,9,10,11,13,14,16,57,58,59,60,61,62,63,68,139,140,143,144,145,147,151,152,154,156,244,245,246,247,248,249,250,251,252,253,254,255,256,258,259,260,262,263,264,265,266,286,287,288,289,359,360,361,362,363,364,365,366,367,368,369,370,371,372,377,383,384,385,386,387,392,394,396,397,399,400,405,408,409]
# index_list = [1,2,3,4,5,6,7,8,9,10,11,139,140,143,144,145,147,151,152,154,156]
# index_list = [13,14,16,58,59,60,61,62,63,68,244,245,246,247,248,249,250,251,252,253,254,255,256,258,259,260,262,263,264,265,266,286,287,288,289,359,360,361,362,363,365,365,366,367,368,369,370,371,372,377,383,384,385,386,387,392,394,396,397,399,400,405,408,409]
# index_list = [154]

# for complex in complex_list:
#     if complex != '3hfm': continue
#     print(f'current is the protein {complex} being tested.')
#     index_list = [390]
    # for i, item in enumerate(combined_items):
    #     if item['pdb'] != complex: continue
    #     mm = item['mut_pos'][0] + item['mut_pos'][-1]
    #     if mm not in cm_list: continue
    #     index_list.append(i)
    # print('index_list:', index_list)

start_time = time.time()
items = []
for item in bench_items:
    index = int(item['No'].split('_')[-1])
    # print('index:', index)
    pdb = item['No'].split('_')[0].upper()
    # if pdb == '3HFM': continue
    # if int(index) <= 100: print('index:', index)
    if (index + 1) in index_list: 
        # print('index:', index)
        items.append(item)
# print('items:', len(items))
# if len(items) == 0: continue

test_loader = DataLoader(items, args.batch_size, args.batch_tokens)
# print()

model.eval()
val_rmsd = []
# 在评估模式下不进行梯度计算
with torch.no_grad():
    for j, batch in enumerate(tqdm(test_loader)):
        # print('j',j)
        proteins = Completize(batch, args.esm)
        X, oS, mS, L, mask, ddG_true = proteins.hbatch
        ddG_pre = model(X, oS, mS, L, mask)
        end_time = time.time()
        predicting_time = end_time - start_time
        print(f'ab-can predicting time: {predicting_time} seconds')

        # print('ddG_pre:', ddG_pre)
        # if torch.isnan(ddG_pre).any():
        #     raise ValueError("Predictions contain NaNs, terminating execution.")
        rmsd = torch.sqrt(torch.sum((ddG_true - ddG_pre)**2)/ddG_pre.shape[0])
        val_rmsd.append(rmsd.item())

        if j == 0:
            predictions = ddG_pre
            trues = ddG_true
        else:
            predictions = torch.cat([predictions, ddG_pre], dim=-1)
            trues = torch.cat([trues, ddG_true], dim=-1)

        # break

    print('predictions:', predictions.shape, predictions)
    print('trues:', trues.shape, trues)
    try: 
        pc, p_value = pearsonr(predictions.cpu(), trues.cpu())
        print(f'rmsd = {sum(val_rmsd) / len(val_rmsd)}, pc = {pc}, p_value = {p_value}')
    except: 
        print('current complex only got one.')

