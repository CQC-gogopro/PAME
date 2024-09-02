import os, json, imageio
import numpy as np
from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import to_cuda, get_output, mkdir_if_missing
import torch
from tqdm import tqdm
from utils.test_utils import test_phase
import torch.distributed as dist
import sys
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def update_tb(tb_writer, tag, loss_dict, iter_no):
    for k, v in loss_dict.items():
        tb_writer.add_scalar(f'{tag}/{k}', v.item(), iter_no)


def train_phase(p, args, train_loader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer, tb_writer_test, iter_count, local_rank, nprocs):
    """ Vanilla training with fixed loss weights """
    model.train() 
    p['num_batches'] = len(train_loader)
    tqdm_train_loader = tqdm(train_loader,file=sys.stdout)
    for i, cpu_batch in enumerate(tqdm_train_loader):
        # Forward pass
        batch = to_cuda(cpu_batch)
        images = batch['image']
        
        if iter_count==0 and p.parameter_analysis:
            # 分析parameters
            parameters = parameter_count_table(model,max_depth=5)
            batchsize = images.shape[0]
            # 分析FLOPs
            # flops = FlopCountAnalysis(model, images)
            # flo = flops.total()*(1.e-9)/batchsize
            if local_rank==0:
                print(f"batch size on one device: {images.shape[0]}, all device num: {nprocs}")
                print(parameters) 
                # print(flops)
                # print('FLOPs:{:.1f}'.format(flo),'G')
        p['iter_count'] = iter_count
        p['epoch'] = epoch
        p['__train__'] = 1
        output = model(images,p)
        iter_count += 1
        
        loss_dict = criterion(output, batch, tasks=p.TASKS.NAMES)
        # get learning rate
        lr = scheduler.get_lr()
        loss_dict['lr'] = torch.tensor(lr[0])
        
        # Backward
        optimizer.zero_grad()

        # print('local_rank',local_rank,"local_dict['total']", loss_dict['total'])
        loss_dict['total'] /= nprocs
        torch.distributed.barrier()
        dist.all_reduce(loss_dict['total'], op=dist.ReduceOp.SUM)
        
        if local_rank == 0:
            tqdm_train_loader.set_description("iters: %i"%(iter_count))
            tqdm_train_loader.set_postfix(loss='{:.2f}'.format(loss_dict['total'].item()),lr='{:.4e}'.format(lr[0]))
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), **p.grad_clip_param)
        
        optimizer.step()
        scheduler.step()

        if iter_count % 400 == 0:
            if 'cv' in loss_dict:
                print("cv:" + str(loss_dict['cv'].item()))
            if 'route' in loss_dict:
                print("route:" + str(loss_dict['route'].item()))

        # end condition
        if iter_count >= p.max_iter:
            print('Max itereaction achieved.')
            # return True, iter_count
            end_signal = True
        else:
            end_signal = False

        # Evaluate
        if end_signal:
            eval_bool = True
        elif iter_count % p.val_interval == 0:
            eval_bool = True 
        else:
            eval_bool = False

        # Perform evaluation
        if eval_bool and args.local_rank == 0:
            print('Evaluate at iter {}'.format(iter_count))
            curr_result = test_phase(p, test_dataloader, model, criterion, iter_count)
            tb_update_perf(p, tb_writer_test, curr_result, iter_count)
            print('Evaluate results at iteration {}: \n'.format(iter_count))
            print(curr_result)
            with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)
            if iter_count%p.save_inter ==0 or end_signal:
                if iter_count >= p.save_after-1:
                    # Checkpoint after evaluation
                    print('Checkpoint starts at iter {}....'.format(iter_count))
                    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(), 
                                'epoch': epoch, 'iter_count': iter_count-1}, p['checkpoint']+str(iter_count))
                    print('Checkpoint finishs.')
            model.train() # set model back to train status

        if end_signal:
            return True, iter_count

    return False, iter_count

# from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]

def tb_update_perf(p, tb_writer_test, curr_result, cur_iter):
    if 'semseg' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/semseg_miou', curr_result['semseg']['mIoU'], cur_iter)
    if 'human_parts' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/human_parts_mIoU', curr_result['human_parts']['mIoU'], cur_iter)
    if 'sal' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/sal_maxF', curr_result['sal']['maxF'], cur_iter)
    if 'edge' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/edge_val_loss', curr_result['edge']['loss'], cur_iter)
    if 'normals' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/normals_mean', curr_result['normals']['mean'], cur_iter)
    if 'depth' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/depth_rmse', curr_result['depth']['rmse'], cur_iter)