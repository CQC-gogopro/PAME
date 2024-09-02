# import zCurve as z 
# code = z.interlace(2,16,8)
import torch
def get_z_order(pic=None ,H=28 , W=36):
    if pic!=None:
        H=pic.shape[-2]
        W=pic.shape[-1]
    result = 0
    shift = 0
    seq = []
    for j in range(H):
        for i in range(W):
            decimal1 = i
            decimal2 = j
            if decimal1 ==1:
                pass
            result = 0
            shift = 0
            while decimal1 > 0 or decimal2 > 0:
                # 取出两个数字的最低位并将其穿插
                bit1 = decimal1 & 1
                bit2 = decimal2 & 1
                result |= (bit1 << (shift * 2)) | (bit2 << (shift * 2 + 1))
                # 将两个数字右移一位，准备处理下一个最低位
                decimal1 >>= 1
                decimal2 >>= 1
                shift += 1
            seq.append(result)
    sample_map = {}
    # result = torch.tensor(seq).view(H,W).t().flatten() # 横向扫描
    result = torch.tensor(seq) # 横向扫描
    _, ind_2sorted = torch.sort(result,dim=0)
    _, ind_2reverse = torch.sort(ind_2sorted,dim=0)
    sample_map['ind_2sorted']=ind_2sorted.cuda()
    sample_map['ind_2reverse']=ind_2reverse.cuda()
    return sample_map  

def get_z_order_4(pic=None ,H=28 , W=36):
    if pic!=None:
        H=pic.shape[-2]
        W=pic.shape[-1]
    result = 0
    shift = 0
    seq = []
    
    max_len = max(H,W)
    for j in range(max_len):
        for i in range(max_len):
            decimal1 = i
            decimal2 = j
            if decimal1 ==1:
                pass
            result = 0
            shift = 0
            while decimal1 > 0 or decimal2 > 0:
                # 取出两个数字的最低位并将其穿插
                bit1 = decimal1 & 1
                bit2 = decimal2 & 1
                result |= (bit1 << (shift * 2)) | (bit2 << (shift * 2 + 1))
                # 将两个数字右移一位，准备处理下一个最低位
                decimal1 >>= 1
                decimal2 >>= 1
                shift += 1
            seq.append(result)
    
    seq_list = []
    seq0 = torch.tensor(seq).view(max_len,max_len)
    seq1 = torch.rot90(seq0, 1, [0, 1])      # 逆时针旋转90
    seq2 = torch.rot90(seq0, 2, [0, 1])      # 逆时针旋转180   
    seq3 = torch.rot90(seq0, 3, [0, 1])      # 逆时针旋转270
    seq_list.append(seq0[:H,:W])
    seq_list.append(seq1[:H,:W])
    seq_list.append(seq2[:H,:W])
    seq_list.append(seq3[:H,:W])
    
    sample_map_list = []
    for r in range(4):
        sample_map = {}
        result = seq_list[r].flatten()  
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
    interleaved_binary = get_z_order(H=H, W=W)
    order_list = get_z_order_4(H=H, W=W)
    a = torch.rand(H*W)
    b = a.index_select(0,interleaved_binary['ind_2sorted'])
    c = b.index_select(0,interleaved_binary['ind_2reverse'])
    print(a)
    print(b)
    print(c)
    '''
    interleaved_binary['ind_2sorted']
                                        tensor([ 0,  1,  5,  6,  2,  3,  7,  8, 10, 11, 15, 16, 12, 13, 17, 18,  4,  9,
                                                14, 19, 20, 21, 25, 26, 22, 23, 27, 28, 30, 31, 35, 36, 32, 33, 37, 38,
                                                24, 29, 34, 39])
    interleaved_binary['ind_2reverse']
                                        tensor([ 0,  1,  5,  6,  2,  3,  7,  8, 10, 11, 15, 16, 12, 13, 17, 18,  4,  9,
                                                14, 19, 20, 21, 25, 26, 22, 23, 27, 28, 30, 31, 35, 36, 32, 33, 37, 38,
                                                24, 29, 34, 39])
    '''
    # f = lambda x,y: 113*x+y
    pass
    # 输出结果
    # print("穿插后的二进制表示为:", interleaved_binary)

