import math
from typing import Iterable, List, Union
import torch
        
def get_s_order(pic=None,H=28,W=36):
    if pic!=None:
        H=pic.shape[-2]
        W=pic.shape[-1]
    order = torch.tensor(range(H*W)).view(H,W)
    for i in range(H):
        if i%2 == 1:
            order[i] = torch.flip(order[i],dims=[0])
    order = order.view(1,-1).squeeze(0)
    sample_map = {}
    _, ind_2sorted = torch.sort(order,dim=0)
    _, ind_2reverse = torch.sort(ind_2sorted,dim=0)
    sample_map['ind_2sorted']=ind_2sorted.cuda()
    sample_map['ind_2reverse']=ind_2reverse.cuda()
    return sample_map

def get_s_order_4(pic=None,H=28,W=36):
    if pic!=None:
        H=pic.shape[-2]
        W=pic.shape[-1]
    M = max(H,W)
    seq_list = []
    seq0 = torch.tensor(range(M*M)).view(M,M)
    for i in range(M):
        if i%2 == 1:
            seq0[i] = torch.flip(seq0[i],dims=[0])

    seq1 = torch.rot90(seq0, 1, [0, 1])      # 逆时针旋转90
    seq2 = torch.rot90(seq0, 2, [0, 1])      # 逆时针旋转180   
    seq3 = torch.rot90(seq0, 3, [0, 1])      # 逆时针旋转270

    seq_list.append(seq0[:H,:W].reshape(1,-1).squeeze(0))
    seq_list.append(seq1[:H,:W].reshape(1,-1).squeeze(0))
    seq_list.append(seq2[:H,:W].reshape(1,-1).squeeze(0))
    seq_list.append(seq3[:H,:W].reshape(1,-1).squeeze(0))

    # for i in range(8):
    #     print(i)
    #     print(seq_list[i].view(H,W))
            
    # seq_list = []
    # seq0 = torch.tensor(seq).view(max_len,max_len).t()
    # seq1 = torch.rot90(seq0, 1, [0, 1])      # 逆时针旋转90
    # seq2 = torch.rot90(seq0, 2, [0, 1])      # 逆时针旋转180   
    # seq3 = torch.rot90(seq0, 3, [0, 1])      # 逆时针旋转270
    # seq_list.append(seq0[:H,:W])
    # seq_list.append(seq1[:H,:W])
    # seq_list.append(seq2[:H,:W])
    # seq_list.append(seq3[:H,:W])
    
    sample_map_list = []
    for r in range(len(seq_list)):
        sample_map = {}
        result = seq_list[r]
        _, ind_2sorted = torch.sort(result,dim=0)
        _, ind_2reverse = torch.sort(ind_2sorted,dim=0)
        sample_map['ind_2sorted']=ind_2sorted.cuda()
        sample_map['ind_2reverse']=ind_2reverse.cuda()
        sample_map_list.append(sample_map)
    return sample_map_list

if __name__ == '__main__':
    # 输入两个十进制数
    # decimal1 = int(input("请输入第一个十进制数："))
    # decimal2 = int(input("请输入第二个十进制数："))
    H = 5
    W = 8
    # 调用函数获得穿插后的二进制结果
    interleaved_binary = get_h_order(H=H, W=W)
    order_list=get_h_order_4(H=H, W=W)
    a = torch.rand(H*W)
    b = a.index_select(0,interleaved_binary['ind_2sorted'])
    c = b.index_select(0,interleaved_binary['ind_2reverse'])
    print(a)
    print(b)
    print(c)
    # f = lambda x,y: 113*x+y
    pass
    # 输出结果
    # print("穿插后的二进制表示为:", interleaved_binary)