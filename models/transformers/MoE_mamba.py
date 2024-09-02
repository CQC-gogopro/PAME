import math
import torch.nn as nn
from einops import repeat
from einops import rearrange as o_rearrange

import torch
import numpy as np
import cv2
from PIL import Image
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from functools import partial
from torch.nn.init import trunc_normal_
# =====================================================
def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()
# =====================================================
def f(module, x):
    return {t: module[t](x[t]) for t in x.keys()}
Ln = partial(nn.LayerNorm, eps=1e-6)
BatchNorm2d = nn.BatchNorm2d
class MoeMamba(nn.Module):
    def __init__(
            self,
            p,
            # basic dims ===========
            d_model=256,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            tasks=[],
            normalized_shape2D=[],
            order_map_list=[],
            rank_list=[16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240],
            Nmoe = 15,
            topk = 9,
            **kwargs,
    ):
        device ='cuda:'+str(p.local_rank)
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.tasks = tasks
        self.order_map_list = order_map_list
        self.rank_list = rank_list
        self.Moe_depth = 1
        self.DiNum = 4
        self.Moe_Num = Nmoe
        self.topk = topk
        self.p = p
        '''
        order_map['ind_2sorted']=ind_2sorted.cuda()
        order_map['ind_2reverse']=ind_2reverse.cuda()
        '''
        device = torch.device("cuda:"+str(p.local_rank))
        if p.promptB == True:
            task_prompt_B = {t: nn.Parameter(torch.zeros(self.d_state)).to(device) for t in self.tasks}
            for task in self.tasks:
                trunc_normal_(task_prompt_B[task], std=0.02)
            setattr(self,'task_prompt_B',task_prompt_B)
        if p.promptC == True:
            task_prompt_C = {t: nn.Parameter(torch.zeros(self.d_state)).to(device) for t in self.tasks}
            for task in self.tasks:
                trunc_normal_(task_prompt_C[task], std=0.02)
            setattr(self,'task_prompt_C',task_prompt_C)
        
        first_decode = nn.Conv2d(p.backbone_dim,d_model,3,1,1)   # nn.Linear(p.backbone_dim,d_model)
        setattr(self,'first_decode',first_decode)
        input_norm = nn.GroupNorm(32,self.d_model)
        setattr(self,'input_norm',input_norm)
        
        if self.p.TaskConv:
            task_conv = nn.ModuleDict({t: nn.Conv2d(self.d_model, self.d_model, 3, 1, 1) for t in self.tasks})
            setattr(self,'task_conv',task_conv)
        
        if self.p.LocalConv:
            task_conv1 = nn.ModuleDict({t: nn.Conv2d(self.d_model, self.d_model, 1,1,0) for t in self.tasks})
            task_norm1 = nn.ModuleDict({t: nn.GroupNorm(32,self.d_model) for t in self.tasks})
            task_conv2 = nn.ModuleDict({t: DWConv(self.d_model, self.d_model, k=5) for t in self.tasks})
            task_norm2 = nn.ModuleDict({t: nn.GroupNorm(32,self.d_model) for t in self.tasks})
            task_conv3 = nn.ModuleDict({t: nn.Conv2d(self.d_model, self.d_model, 1,1,0) for t in self.tasks})
            task_norm3 = nn.ModuleDict({t: nn.GroupNorm(32,self.d_model) for t in self.tasks})
            setattr(self,'task_conv1',task_conv1)
            setattr(self,'task_norm1',task_norm1)
            setattr(self,'task_conv2',task_conv2)
            setattr(self,'task_norm2',task_norm2)
            setattr(self,'task_conv3',task_conv3)
            setattr(self,'task_norm3',task_norm3)
        
        act = nn.SiLU()
        setattr(self,'act',act)
        if self.p.SSM == True:
            for d in range(self.Moe_depth):
                im_size = len(self.order_map_list[d%len(self.order_map_list)]['ind_2sorted'])
                # expand_dim:  conv2d; 输入变换 =====================
                # input_norm = nn.ModuleDict({t: BatchNorm2d(self.d_model) for t in self.tasks})
                # dim_expand_proj_xz = nn.ModuleDict({t: nn.Conv2d(self.d_model, self.d_inner*2, kernel_size=1, bias=False) for t in self.tasks})
                task_input_norm = nn.ModuleDict({t: nn.GroupNorm(32,self.d_model) for t in self.tasks})
                setattr(self,f'task_input_norm{d}',task_input_norm)
                #TODO proj_z = MoE(self.d_model, self.d_model, kernal=1, topk=self.topk, ranklist=rank_list, im_size=im_size, tasks=self.tasks) 
                proj_z = nn.ModuleDict({t: nn.Conv2d(self.d_model,self.d_model,1,1,0) for t in self.tasks})
                #TODO dim_expand_proj_x = MoE(self.d_model, self.d_inner, kernal=1, topk=self.topk, ranklist=rank_list, im_size=im_size, tasks=self.tasks) 
                if 'dwconv' in p.keys() and p['dwconv']:
                    conv2d = nn.ModuleDict({t: DWConv(self.d_model,self.d_model,) for t in self.tasks})
                else:
                    conv2d = MoE(p,self.d_model, self.d_model, kernal=3, topk=self.topk, ranklist=rank_list, im_size=im_size, tasks=self.tasks, show_expert_gate=False, name='L') 
                    
                dim_expand_proj_x = nn.ModuleDict({t: nn.Conv2d(self.d_model,self.d_inner,1,1,0) for t in self.tasks})
                # conv2d = nn.ModuleDict({t: DWConv(self.d_inner,self.d_inner,) for t in self.tasks})
                # conv2d = nn.ModuleDict({t: nn.Conv2d(self.d_inner,self.d_inner,3,1,1) for t in self.tasks})
                
                # 计算多任务ABC ====================================
                A_log = {t: self.A_log_init(self.d_state, self.d_inner,device=device) for t in self.tasks} # (D, N)
                D = {t: self.D_init(self.d_inner,device=device) for t in self.tasks}  # (D)
                dt_proj = nn.ModuleDict({t: self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,**factory_kwargs) for t in self.tasks})
                x2dt_proj = nn.ModuleDict({t: nn.Linear(self.d_inner, self.dt_rank, bias=False, **factory_kwargs) for t in self.tasks})
                # 输出层 =========================================fusion
                # dim_decrease_proj = nn.ModuleDict({t: nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs) for t in self.tasks})
                # direction_norm = nn.ModuleDict({t: BatchNorm2d(self.d_inner) for t in self.tasks})
                #TODO dim_decrease_proj = MoE(self.d_inner, self.d_model, kernal=1, topk=self.topk, ranklist=rank_list, im_size=im_size, tasks=self.tasks)  
                dim_decrease_proj = nn.ModuleDict({t: nn.Conv2d(self.d_inner,self.d_model,1,1,0) for t in self.tasks})
                # output_norm = nn.ModuleDict({t: BatchNorm2d(self.d_model) for t in self.tasks})         
                
                #* shared||
                #* MoE BC ===================================================
                if self.p.MoE_C == True:
                    C_proj = MoE(p,self.d_inner, self.d_state, self.topk, ranklist=[4]*self.Moe_Num, im_size=im_size, tasks=self.tasks, show_expert_gate=False, name='C')
                else:
                    C_proj = nn.ModuleDict({t: nn.Conv2d(self.d_inner, self.d_state, 1,1,0) for t in self.tasks})
                if self.p.MoE_B == True:
                    B_proj = MoE(p,self.d_inner, self.d_state, self.topk, ranklist=[4]*self.Moe_Num, im_size=im_size, tasks=self.tasks, show_expert_gate=False, name='B')
                else:
                    B_proj = nn.ModuleDict({t: nn.Conv2d(self.d_inner, self.d_state, 1,1,0) for t in self.tasks})
                
                # 设置属性 ========================================fusion
                # setattr(self,f'input_norm{d}',input_norm)
                setattr(self,f'proj_z{d}',proj_z)
                setattr(self,f'conv2d{d}',conv2d)
                setattr(self,f'dim_expand_proj_x{d}',dim_expand_proj_x)
                setattr(self,f'A_log{d}',A_log)
                setattr(self,f'D{d}',D)
                setattr(self,f'dt_proj{d}',dt_proj)
                setattr(self,f'x2dt_proj{d}',x2dt_proj)
                setattr(self,f'C_proj{d}',C_proj)
                setattr(self,f'B_proj{d}',B_proj)
                
                # setattr(self,f'direction_norm{d}',direction_norm)
                setattr(self,f'dim_decrease_proj{d}',dim_decrease_proj)
                # setattr(self,f'output_norm{d}',output_norm)
            
        self.to(device)
        

    def forward(self, fea_stage, H=28, W=36):
        """
        Args:
            fea_stage (BCHW): a feature from one bacbone stage

        Returns:
            tasks_fea {t: BCHW}: task feature
        """
        Batch, L, dim = fea_stage.shape    # b l d
        
        first_decode = getattr(self,'first_decode')
        input_norm = getattr(self,'input_norm')
        if self.p.TaskConv:
            task_conv = getattr(self,'task_conv')
        if self.p.LocalConv:
            task_conv1 = getattr(self,'task_conv1')
            task_norm1 = getattr(self,'task_norm1')
            task_conv2 = getattr(self,'task_conv2')
            task_norm2 = getattr(self,'task_norm2')
            task_conv3 = getattr(self,'task_conv3')
            task_norm3 = getattr(self,'task_norm3')
        act = getattr(self,'act')
        
        #* 维度变换
        fea_stage = rearrange(fea_stage, "b (h w) d-> b d h w",h=H,w=W)   
        task_fea = input_norm(first_decode(fea_stage))
        #* 卷积分支
        if self.p.LocalConv:
            conv_task_fea = {t: act(task_norm1[t](task_conv1[t](task_fea.clone()))) for t in self.tasks}
            conv_task_fea = {t: act(task_norm2[t](task_conv2[t](conv_task_fea[t]))) for t in self.tasks}
            conv_task_fea = {t: act(task_norm3[t](task_conv3[t](conv_task_fea[t]))) for t in self.tasks}
        #* MLoSSM分支
        if self.p.TaskConv:
            task_fea = {t: task_conv[t](task_fea.clone()) for t in self.tasks}
        else: 
            task_fea = {t: task_fea.clone() for t in self.tasks}
        #* fusion

        if self.p.SSM == False:
            task_fea = {t: task_fea[t] + conv_task_fea[t] for t in self.tasks}
            return task_fea
        for d in range(self.Moe_depth):
            #!================================================================================
            #* 获取网络权重 ===================================fusion
            # input_norm = getattr(self,f'input_norm{d}')
            # input_norm = getattr(self,f'input_norm{d}')
            task_input_norm = getattr(self,f'task_input_norm{d}')
            proj_z = getattr(self,f'proj_z{d}')
            conv2d = getattr(self,f'conv2d{d}')
            dim_expand_proj_x = getattr(self,f'dim_expand_proj_x{d}')
            A_log = getattr(self,f'A_log{d}')
            D = getattr(self,f'D{d}')
            dt_proj = getattr(self,f'dt_proj{d}')
            x2dt_proj = getattr(self,f'x2dt_proj{d}')
            
            C_proj = getattr(self,f'C_proj{d}')
            B_proj = getattr(self,f'B_proj{d}')

            
            
            
            # direction_norm = getattr(self,f'direction_norm{d}')
            dim_decrease_proj = getattr(self,f'dim_decrease_proj{d}')
            #!================================================================================
            # 任务卷积
            # task_fea = {t: task_conv[t](task_fea[t]) for t in self.tasks}
            # task_fea = {t: rearrange(input_norm[t](task_fea[t]), "b (h w) d-> b d h w",h=H,w=W)  for t in self.tasks}
            #* 获得z ==============================MOE
            shortcut = {t: task_fea[t].clone() for t in self.tasks}
            # task_fea = {t: input_norm[t](task_fea[t]) for t in self.tasks}
            #TODO task_z = proj_z(task_fea)
            task_fea = {t: task_input_norm[t](task_fea[t]) for t in self.tasks}
            task_z = {t: act(proj_z[t](task_fea[t])) for t in self.tasks}
            #* 输入卷积 ============================MOE
            if 'dwconv' in self.p.keys() and self.p['dwconv']:
                task_x = {t: conv2d[t](task_fea[t]) for t in self.tasks}
            else:
                task_x = conv2d(task_fea)  
            
            task_x = {t: act(task_fea[t]) for t in self.tasks}
            #* 输入扩张 ===============================MOE
            #TODO task_x = dim_expand_proj_x(task_x)  # 通道expand    获得扩张x
            task_x = {t: dim_expand_proj_x[t](task_x[t]) for t in self.tasks}  # 通道expand    获得扩张x
            del task_fea
            #* 获得dt =================================
            _task_x = {t: rearrange(task_x[t].clone(), "b d h w-> (b h w) d",h=H,w=W) for t in self.tasks}
            dt = {t: dt_proj[t].weight @ x2dt_proj[t](_task_x[t]).t() for t in self.tasks}  # (bl d)
            dt = {t: rearrange(dt[t], "d (b l) -> b d l", l=L).contiguous() for t in self.tasks}
            del _task_x
            
            #* B和C ===================================MOE
            if self.p.MoE_B:
                B = B_proj(task_x)    # b d h w
            else:
                B = {t: B_proj[t](task_x[t]) for t in self.tasks}
            if self.p.MoE_C:
                C = C_proj(task_x)    # b d h w
            else:
                C = {t: C_proj[t](task_x[t]) for t in self.tasks}
            B = {t: rearrange(B[t], " b d h w -> b d (h w)") for t in self.tasks}
            C = {t: rearrange(C[t], " b d h w -> b d (h w)") for t in self.tasks}
            task_x = {t: rearrange(task_x[t], "b d h w -> b d (h w)") for t in self.tasks}
            #* 对输入的x、B、C、dt重排
            # task_x = [{t: torch.index_select(task_x[t], 2, self.order_map_list[(d*self.DiNum + i)%len(self.order_map_list)]['ind_2sorted']) for t in self.tasks} for i in range(self.DiNum)]
            # B = [{t: torch.index_select(B[t], 2, self.order_map_list[(d*self.DiNum + i)%len(self.order_map_list)]['ind_2sorted']) for t in self.tasks} for i in range(self.DiNum)]
            # C = [{t: torch.index_select(C[t], 2, self.order_map_list[(d*self.DiNum + i)%len(self.order_map_list)]['ind_2sorted']) for t in self.tasks} for i in range(self.DiNum)]
            # dt = [{t: torch.index_select(dt[t], 2, self.order_map_list[(d*self.DiNum + i)%len(self.order_map_list)]['ind_2sorted']) for t in self.tasks} for i in range(self.DiNum)]
            for task in self.tasks:
                if self.p.promptB == True:
                    task_prompt_B = getattr(self,f'task_prompt_B')
                    B[task] += task_prompt_B[task][None,:,None].expand(Batch,-1,L).contiguous()
                if self.p.promptC == True:
                    task_prompt_C = getattr(self,f'task_prompt_C')
                    C[task] += task_prompt_C[task][None,:,None].expand(Batch,-1,L).contiguous()
            task_x = [{t: task_x[t][:,:,self.order_map_list[(d*self.DiNum + i)%len(self.order_map_list)]['ind_2sorted']] for t in self.tasks} for i in range(self.DiNum)]
            B = [{t: B[t][:,:,self.order_map_list[(d*self.DiNum + i)%len(self.order_map_list)]['ind_2sorted']] for t in self.tasks} for i in range(self.DiNum)]
            C = [{t: C[t][:,:,self.order_map_list[(d*self.DiNum + i)%len(self.order_map_list)]['ind_2sorted']] for t in self.tasks} for i in range(self.DiNum)]
            dt = [{t: dt[t][:,:,self.order_map_list[(d*self.DiNum + i)%len(self.order_map_list)]['ind_2sorted']] for t in self.tasks} for i in range(self.DiNum)]
            
            
            #* 获得A ======================================
            A = {t: -torch.exp(A_log[t].float()) for t in self.tasks}  # (k * d, d_state)
            
            #* SSM =======================================
            fea = {t:0 for t in self.tasks}
            for task in self.tasks:
                for di in range(self.DiNum):
                    feature = selective_scan_fn(
                        task_x[di][task], dt[di][task],
                        A[task], B[di][task], C[di][task], D[task].float(),
                        delta_bias=dt_proj[task].bias.float(),
                        delta_softplus=True,
                    )
                    #* 还原排序 =============================
                    # feature = torch.index_select(feature, 2, self.order_map_list[(d*self.DiNum + di)%len(self.order_map_list)]['ind_2reverse'])
                    feature = feature[:,:,self.order_map_list[(d*self.DiNum + di)%len(self.order_map_list)]['ind_2reverse']]
                    fea[task] += feature
            
            #* 输出变换 =======================================MOE
            fea = {t: rearrange(fea[t], "b d (h w)-> b d h w",h=H,w=W)  for t in self.tasks}
            
            #TODO task_fea = dim_decrease_proj(fea)
            task_fea = {t: dim_decrease_proj[t](fea[t]) for t in self.tasks}
            for task in self.tasks:
                task_fea[task] = task_fea[task] * task_z[task]
            if self.p.LocalConv:
                task_fea = {t: shortcut[t] + task_fea[t] + conv_task_fea[t] for t in self.tasks}
            else:
                task_fea = {t: shortcut[t] + task_fea[t] for t in self.tasks}
            
        return task_fea
    
    
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Ou
        # r initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        return x * self.sigmoid(attn)


class ChannelAttentionBlock(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=16):
        super(ChannelAttentionBlock, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
  
class DWConv(nn.Module):
    def __init__(self,in_channel, out_channel, dilation=1, stride=1, padding=1, k=3):
        super().__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=k,
                                    stride=stride,
                                    padding=k//2,
                                    groups=in_channel,
                                    dilation=dilation)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
        #逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    dilation=dilation)
        self.norm = Ln(out_channel)
        self.act = nn.SiLU()
        # self.to(torch.device("cuda:0"))
    def forward(self,x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        out = self.norm(out.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()
        out = self.act(out)
        return out
    

class SpatialAtt(nn.Module):
    def __init__(self, dim, dim_out, im_size, with_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim_out)
        self.convsp = nn.Linear(im_size, 1)
        self.ln_sp = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.conv3 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.with_feat = with_feat
        if with_feat:
            self.feat_linear = nn.Conv2d(dim_out *2 , dim_out *2, kernel_size=1)
    
    def forward(self, x):
        n, _, h, w = x.shape
        feat = self.conv1(x)
        feat = self.ln(feat.reshape(n, -1, h * w).permute(0, 2, 1).contiguous()).permute(0, 2, 1).reshape(n, -1, h, w).contiguous()
        feat = self.act(feat)
        feat = self.conv3(feat)

        feat_sp = self.convsp(x.reshape(n, -1, h * w)).reshape(n, 1, -1)
        feat_sp = self.ln_sp(feat_sp).reshape(n, -1, 1, 1)
        feat_sp = self.act(feat_sp)
        feat_sp = self.conv2(feat_sp)
        
        n, c, h, w = feat.shape
        feat = torch.mean(feat.reshape(n, c, h * w), dim=2).reshape(n, c, 1, 1)
        feat = torch.cat([feat, feat_sp], dim=1)

        return feat

class MoE(nn.Module):
    def __init__(self,p, in_channels, out_channels, kernal=1, topk=9, ranklist=[], im_size=0, tasks=[], show_expert_gate=False, name=''):
        super().__init__()
        #* task specific  router
        self.tasks = tasks
        self.show_expert_gate = show_expert_gate
        self.name = name
        Nmoe = len(ranklist)
        # if p.SimpleRouter:
        #     self.router = nn.ModuleDict({t: SimpleRouter(p.H*p.W*in_channels/(2**(p.SpacePoolingRate+1)*2**p.DimPoolingRate),
        #                                                  Nmoe)
        #                             for t in self.tasks}) 
        # else:
        #     self.router = nn.ModuleDict({t: nn.Sequential(
        #                                     SpatialAtt(in_channels, in_channels // 4, im_size=im_size, with_feat=False),
        #                                     nn.Conv2d(in_channels // 2, Nmoe * 2 + 1, kernel_size=1)) 
        #                             for t in self.tasks}) 
        self.router = nn.ModuleDict({t: nn.Sequential(
                                            SpatialAtt(in_channels, in_channels // 4, im_size=im_size, with_feat=False),
                                            nn.Conv2d(in_channels // 2, Nmoe * 2 + 1, kernel_size=1)) 
                                    for t in self.tasks}) 
        self.topk = topk
        self.moe_num = Nmoe
        self.desert_k = Nmoe - self.topk    # 这些MoE项的输出会被置0
        self.im_size = im_size
        
        self.Bn = nn.ModuleDict({t: BatchNorm2d(out_channels) for t in self.tasks})
        self.Bn_all = nn.ModuleDict({t: BatchNorm2d(out_channels) for t in self.tasks})
        
        padding = kernal//2
        for r in [ranklist[0]]:
            if r > 0:
                self.proj_task = nn.ModuleDict({t: LoraConv(in_channels, out_channels,kernal,1,r,padding) for t in self.tasks})
            else: 
                self.proj_task = nn.ModuleDict({t: nn.Conv2d(in_channels, out_channels,kernal,1,padding) for t in self.tasks})
        self.proj_gerneral = nn.Conv2d(in_channels, out_channels,kernal,1,padding)
        self.MoE = nn.ModuleList()
        for r in ranklist:
            if r > 0:
                self.MoE.append(LoraConv(in_channels, out_channels,kernal,1,r,padding))
            else: 
                self.MoE.append(nn.Conv2d(in_channels, out_channels,kernal,1,padding))

    def forward(self,task_x):
        Batch, Channel, H, W = task_x['semseg'].shape
        
        router_probs = {t: self.router[t](task_x[t]) for t in self.tasks}
        prob_lora = {t: router_probs[t][:, :self.moe_num * 2].chunk(2, dim=1) for t in self.tasks}    # [b 2Moe 1 1]*2 
        prob_mix = {t: router_probs[t][:, self.moe_num * 2:] for t in self.tasks}    # b 1 1 1 
        
        route_raw = {t: prob_lora[t][0] for t in self.tasks}    # b Moe 1 1
        stdev = {t: prob_lora[t][1] for t in self.tasks} # b, Moe, 1, 1
        if self.training:
            noise = {t: torch.randn_like(route_raw[t]) * stdev[t] for t in self.tasks}
        else:
            noise = {t: 0 for t in self.tasks}

        route_raw = {t: torch.softmax(route_raw[t] + noise[t], dim=1) for t in self.tasks}
        route_indice = {t: torch.topk(route_raw[t], self.desert_k, dim=1, largest=False)[1] for t in self.tasks}
        route_1 = {t: route_raw[t].clone() for t in self.tasks}
        for j in range(Batch):
            for i in range(self.desert_k):
                for t in self.tasks:
                    route_1[t][j, route_indice[t][j, i].reshape(-1)] = 0
        lora_out = {t: [] for t in self.tasks}
        for i in range(self.moe_num):
            for t in self.tasks:
                lora_out[t].append(self.MoE[i](task_x[t]).unsqueeze(1)) # n, 1, c, h, w
        out = {}
        for t in self.tasks:
            lora_out[t] = torch.sum( torch.cat(lora_out[t], dim=1) * route_1[t].unsqueeze(4) ,  dim=1)
            lora_out[t] = self.Bn_all[t](lora_out[t])
            out[t] = lora_out[t] + self.proj_task[t](task_x[t]) * prob_mix[t][:, 0].unsqueeze(3) + self.proj_gerneral(task_x[t].detach())
            out[t] = self.Bn[t](out[t])
            
        if self.show_expert_gate:
            result = []
            for t in self.tasks:
                result.append(route_1[t].cpu().numpy()[0,...])
            result = np.array(result)
            shape = 20
            result = np.repeat(np.repeat(result, shape, axis=0), shape, axis=1)
            result = ((1-result) * 255).astype(np.uint8)
            result = np.squeeze(result)
            im = cv2.applyColorMap(result, cv2.COLORMAP_BONE) 
            im=Image.fromarray(im)
            im.save('/18179377869/code/Multi-Task-Transformer/yxz/work_dir/MoE_gate_vision/'+self.name+'.png')
        return out # b c h w

class LoraConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,stride=1, rank=6, padding=0):
        super().__init__()
        self.W = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.M = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)
        self.init_weights()
        
    def init_weights(self):
        nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W.bias)
        nn.init.kaiming_uniform_(self.M.weight, a=math.sqrt(5))
        nn.init.zeros_(self.M.bias)
    
    def forward(self, x):
        x = self.W(x)
        x = self.M(x)
        return x
    
    
class InputPooling(nn.Module):
    def __init__(self,space=2,channel=4):
        super(InputPooling, self).__init__()
        self.space = space # 空间缩减指数
        self.channel = channel # 通道缩减指数
        # 定义空间池化操作
        self.spatial_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.spatial_max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 定义组合的1D池化操作
        self.avg_pool1d = nn.AvgPool1d(kernel_size=2, stride=2)
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): BCHW

        Returns:
            tensor: B C/16 H/2 W/2
        """
        # 先缩减空间尺寸
        for _ in range(self.space):
            x_avg = self.spatial_avg_pool(x)
            x_max = self.spatial_max_pool(x)
            x = (x_avg + x_max) / 2
        
        # 转换形状以适应1D池化操作
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w-> b (h w) c")        
        
        # 进行四次组合的1D池化操作，取平均
        for _ in range(self.channel):
            x_avg = self.avg_pool1d(x)
            x_max = self.max_pool1d(x)
            x = (x_avg + x_max) / 2

        # x = rearrange(x, "b (h w) c->b c h w",h=H,w=W)
        
        return x

class SimpleRouter(nn.Module):
    def __init__(self,in_dim,out_dim):
        """_summary_

        Args:
            in_dim (_type_): router的映射输入维度
            out_dim (_type_): 专家数
        """
        super(SimpleRouter, self).__init__()
        self.pooling = InputPooling(2,4)
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(in_dim)
        # self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        b, c ,h ,w = x.shape
        x = self.pooling(x)
        x = rearrange(x, "b l c->b (l c)")
        x = self.norm(x)
        prob = self.linear(x)
        prob = torch.unsqueeze(prob, -1)
        prob = torch.unsqueeze(prob, -1)
        return prob