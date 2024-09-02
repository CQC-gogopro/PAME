from evaluation.evaluate_utils import PerformanceMeter
from tqdm import tqdm
from utils.utils import get_output, mkdir_if_missing
from evaluation.evaluate_utils import save_model_pred_for_one_task
import torch
import os

@torch.no_grad()
def test_phase(p, test_loader, model, criterion, epoch):
    all_tasks = [t for t in p.TASKS.NAMES]
    two_d_tasks = [t for t in p.TASKS.NAMES]
    performance_meter = PerformanceMeter(p, two_d_tasks)

    model.eval()
    p['__train__'] = 0
    tasks_to_save = []
    # # if 'depth' in all_tasks:
    # #     tasks_to_save.append('depth')
    # # if 'normals' in all_tasks:
    # #     tasks_to_save.append('normals')
    # if 'edge' in all_tasks:
    #     tasks_to_save.append('edge')
    tasks_to_save = all_tasks

    if p['pred_vis'] == True:
        pred_save_dirs = {task: os.path.join(p['save_dir'], 'pred', task) for task in tasks_to_save}
        p['pred_save_dirs'] = pred_save_dirs
        for save_dir in pred_save_dirs.values():
            mkdir_if_missing(save_dir)
    else:
        p['pred_save_dirs'] = None
        
    if p['head_vis_tsne'] == True:  
        head_tsne_save_dirs = {task: os.path.join(p['save_dir'], 'head_tsne', task) for task in tasks_to_save}
        p['head_tsne_save_dirs'] = head_tsne_save_dirs
        for save_dir in head_tsne_save_dirs.values():
            mkdir_if_missing(save_dir)
    else:
        p['head_tsne_save_dirs'] = None
        
    if p['head_vis_pca'] == True:
        head_pca_save_dirs = {task: os.path.join(p['save_dir'], 'head_pca', task) for task in tasks_to_save}
        p['head_pca_save_dirs'] = head_pca_save_dirs
        for save_dir in head_pca_save_dirs.values():
            mkdir_if_missing(save_dir)
    else:
        p['head_pca_save_dirs'] = None
        
    if p['backbone_stage3_vis'] == True:
        head_pca_save_dirs = os.path.join(p['save_dir'], 'backbone_stage3_pca')
        p['backbone_stage3_dirs'] = head_pca_save_dirs
        mkdir_if_missing(head_pca_save_dirs)
    else:
        p['backbone_stage3_dirs'] = None
    
            
    route_1_task_sum = []
    route_2_task_sum = []
    for i in range(4):
        route_1_task_sum.append(0)
        route_2_task_sum.append(0)
    route_1_task = []
    route_2_task = []
    for i in range(4):
        route_1_task.append({task:0 for task in p.TASKS.NAMES})
        route_2_task.append({task:0 for task in p.TASKS.NAMES})
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        with torch.no_grad():
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in two_d_tasks}
            p['meta'] = batch['meta']
            model.eval()
            # pp=0
            # for i in range(6):
            #     if '0562' == p['meta']['img_name'][i]:
            #         print(1)
            #         pp=1
            #         break
            # if pp==0:
            #     continue
                
            output = model.module(images,p) # to make ddp happy
            
        
            # Measure loss and performance
            performance_meter.update({t: get_output(output[t], t) for t in two_d_tasks}, 
                                    {t: targets[t] for t in two_d_tasks})
            if p['pred_vis'] == True:
                for task in tasks_to_save:
                    
                    save_model_pred_for_one_task(p, i, batch, output, p['pred_save_dirs'], task, epoch=epoch)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results


@torch.no_grad()
def vis_phase(p, test_loader, model, criterion, epoch):
    from utils.visualization_utils import vis_pred_for_one_task, visulization_for_gt
    # Originally designed for visualization on cityscapes-3D
    model.eval()

    tasks_to_save = ['semseg', 'human_parts', 'sal','edge','normals']

    save_dirs = {task: os.path.join(p['save_dir'], 'vis', task) for task in tasks_to_save}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        p['__train__'] = 1
            
        if 'dataset_vis' in p.keys() and p['dataset_vis']:  # 数据集的可视化
            for task in tasks_to_save:
                visulization_for_gt(p, batch, save_dirs[task], task)
        else:
            output = model.module(images, p) # to make ddp happy
            for task in tasks_to_save:
                vis_pred_for_one_task(p, batch, output, save_dirs[task], task)
            del output

        
        del batch, images
    return

