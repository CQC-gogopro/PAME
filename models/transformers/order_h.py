from hilbertcurve.hilbertcurve import HilbertCurve
import math
from typing import Iterable, List, Union
import torch
class Horder():
    def __init__(self,H=28,W=36):
        p = math.ceil(math.log(max(H,W),2))
        self.p = p
        self.n = 2
    def distance_from_point(self, point) -> int:
        point = [int(el) for el in point]
        m = 1 << (self.p - 1)
        # Inverse undo excess work
        q = m
        while q > 1:
            p = q - 1
            for i in range(self.n):
                if point[i] & q:
                    point[0] ^= p
                else:
                    t = (point[0] ^ point[i]) & p
                    point[0] ^= t
                    point[i] ^= t
            q >>= 1

        # Gray encode
        for i in range(1, self.n):
            point[i] ^= point[i-1]
        t = 0
        q = m
        while q > 1:
            if point[self.n-1] & q:
                t ^= q - 1
            q >>= 1
        for i in range(self.n):
            point[i] ^= t

        distance = self._transpose_to_hilbert_integer(point)
        return distance
    def _transpose_to_hilbert_integer(self, x: Iterable[int]) -> int:
        """Restore a hilbert integer (`h`) from its transpose (`x`).

        Args:
            x (list): transpose of h
                (n components with values between 0 and 2**p-1)

        Returns:
            h (int): integer distance along hilbert curve
        """
        x_bit_str = [self._binary_repr(x[i], self.p) for i in range(self.n)]
        h = int(''.join([y[i] for i in range(self.p) for y in x_bit_str]), 2)
        return h
    def _binary_repr(self,num: int, width: int) -> str:
        """Return a binary string representation of `num` zero padded to `width`
        bits."""
        return format(num, 'b').zfill(width)
def get_h_order(pic=None,H=28,W=36):
    if pic!=None:
        H=pic.shape[-2]
        W=pic.shape[-1]
    h_order = Horder(H=H, W=W)
    seq = []
    for i in range(H):
        for j in range(W):
            seq.append(h_order.distance_from_point([i,j]))
    sample_map = {}
    result = torch.tensor(seq).view(H,W).t().flatten() # 横向扫描
    _, ind_2sorted = torch.sort(result,dim=0)
    _, ind_2reverse = torch.sort(ind_2sorted,dim=0)
    sample_map['ind_2sorted']=ind_2sorted.cuda()
    sample_map['ind_2reverse']=ind_2reverse.cuda()
    return sample_map

def get_h_order_4(pic=None,H=28,W=36):
    if pic!=None:
        H=pic.shape[-2]
        W=pic.shape[-1]
    h_order = Horder(H=H, W=W)
    seq = []
    max_len = 2**h_order.p
    for i in range(max_len):
        for j in range(max_len):
            seq.append(h_order.distance_from_point([i,j]))
            
    seq_list = []
    seq0 = torch.tensor(seq).view(max_len,max_len).t()
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

def get_h_order_8(pic=None,H=28,W=36):
    if pic!=None:
        H=pic.shape[-2]
        W=pic.shape[-1]
    h_order = Horder(H=H, W=W)
    seq = []
    max_len = 2**h_order.p
    for i in range(max_len):
        for j in range(max_len):
            seq.append(h_order.distance_from_point([i,j]))
            
    seq_list = []
    seq0 = torch.tensor(seq).view(max_len,max_len).t()
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
    for r in range(4):
        sample_map = {}
        result = torch.flip(seq_list[r].flatten(),dims=[0])
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
    order_list=get_h_order_8(H=H, W=W)
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