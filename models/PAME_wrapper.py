import torch.nn as nn
import torch.nn.functional as F

INTERPOLATE_MODE = 'bilinear'

class PAMEWrapper(nn.Module):
    def __init__(self, p, backbone, heads, aux_heads=None):
        super(PAMEWrapper, self).__init__()
        self.tasks = p.TASKS.NAMES

        self.backbone = backbone
        self.heads = heads 

    def forward(self, x, p, need_info=False):
        img_size = x.size()[-2:]
        # print("iter:",p['iter_count'],"batch:",x[0,0:5,100,100])  # 测试相同的seed会不会导致不同卡上相同的内容
        # print("iter:",p['iter_count'],"parameter",self.heads['semseg'].linear_pred.weight[0,0])
        out = {}

        target_size = img_size

        # TaskPrompter 
        task_features, info = self.backbone(x, p) 

        # Generate predictions
        out = {}
        for t in self.tasks:
            if t in task_features:
                _task_fea = task_features[t]
                if not p['__train__']:
                    info = {'task':t,
                            'meta':p['meta'],
                            'head_tsne_save_dirs':p['head_tsne_save_dirs'],
                            'head_pca_save_dirs':p['head_pca_save_dirs'],
                            'head_vis_pca_reverse':p['head_vis_pca_reverse']}
                info['__train__'] = p['__train__']
                info['im_size'] = (p.H,p.W)
                out[t] = F.interpolate(self.heads[t](_task_fea, info), target_size, mode=INTERPOLATE_MODE)
        for key in info.keys():
            if 'route' in key:
                out[key] = info[key]
        if need_info:
            return out, info
        else:
            return out