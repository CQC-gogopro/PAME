import argparse
import cv2
import os
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
import numpy as np
import sys
import torch
import pdb

from utils.utils import mkdir_if_missing
from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_test_dataset, get_train_dataloader, get_test_dataloader,\
                                get_optimizer, get_model, get_criterion
from utils.logger import Logger
from utils.train_utils import train_phase
from utils.test_utils import test_phase, vis_phase
from evaluation.evaluate_utils import PerformanceMeter
import torch.nn as nn

# import wandb

from torch.utils.tensorboard import SummaryWriter
import time
start_time = time.time()

# DDP
import torch.distributed as dist
import datetime
import torch.multiprocessing as mp
import random

# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--run_mode',
                    help='Config file for the experiment')
parser.add_argument('--trained_model', default=None,
                    help='Config file for the experiment')
args = parser.parse_args()

# CUDNN
torch.backends.cudnn.benchmark = True
# opencv
cv2.setNumThreads(0)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main_worker(local_rank, nprocs, args): 
    args.local_rank = local_rank
    print('local rank: %s' %args.local_rank)
    torch.cuda.set_device(args.local_rank) 
    # set_seed(0)
    set_seed(int(args.netport)+int(local_rank))
    # Retrieve config file
    params = {'run_mode': args.run_mode}

    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:'+args.netport,
                            world_size=args.nprocs, 
                            rank=local_rank,
                            timeout=datetime.timedelta(seconds=7200))
    p = create_config(args.config_exp, params)
    p.local_rank = local_rank
    if args.local_rank == 0:
        log_dir = os.path.join(p['output_dir'], 'log_file'+ time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) +'.log')
        sys.stdout = Logger(log_dir)
        print('log dir: ',log_dir)
        print(p)
    # tensorboard
    tb_log_dir = p.root_dir + '/tb_dir'
    p.tb_log_dir = tb_log_dir
    if args.local_rank == 0:
        train_tb_log_dir = tb_log_dir + '/train'
        test_tb_log_dir = tb_log_dir + '/test'
        if args.run_mode != 'infer':
            mkdir_if_missing(tb_log_dir)
            mkdir_if_missing(train_tb_log_dir)
            mkdir_if_missing(test_tb_log_dir)
            # os.environ["WANDB_SILENT"] = "true"
            # wandb.init(project="taskprompter", name=p['version_name'], sync_tensorboard=True, dir=os.path.abspath(tb_log_dir))
        tb_writer_train = SummaryWriter(train_tb_log_dir)
        tb_writer_test = SummaryWriter(test_tb_log_dir)
        print(f"Tensorboard dir: {tb_log_dir}")
    else:
        tb_writer_train = None
        tb_writer_test = None


    # Get model
    model = get_model(p)
    if local_rank==0 and 0:
        print(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    # model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    if True:
        total_params = sum(par.numel() for par in model.parameters())
        total_trainable_params = sum(par.numel() for par in model.parameters() if par.requires_grad)
        if args.local_rank == 0:
            print(f'{total_params:,} total parameters.')
            print(f'{total_trainable_params:,} training parameters.')
    # Get criterion
    criterion = get_criterion(p).cuda()

    # Optimizer
    scheduler, optimizer = get_optimizer(p, model)

    # Performance meter init
    PerformanceMeter(p, [t for t in p.TASKS.NAMES])

    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    if args.run_mode != 'infer':
        train_dataset = get_train_dataset(p, train_transforms)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        train_dataloader = get_train_dataloader(p, train_dataset, train_sampler)
    test_dataset = get_test_dataset(p, val_transforms)
    test_dataloader = get_test_dataloader(p, test_dataset)
    
    DEBUG_FLAG = 0
    if DEBUG_FLAG:
        if args.local_rank == 0:
            print("\nFirst Testing...")
        eval_test = test_phase(p, test_dataloader, model, criterion, 0)
        if args.local_rank == 0:
            print(eval_test)
    
    
    
    # Train loop
    if args.run_mode != 'infer':
        # Resume from checkpoint
        if os.path.exists(p['checkpoint']):
            if args.trained_model != None:
                checkpoint_path = args.trained_model
            else:
                checkpoint_path = p['checkpoint']
            if args.local_rank == 0:
                print('Use checkpoint {}'.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint.keys():
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint.keys():
                start_epoch = checkpoint['epoch'] + 1 # epoch count is not used
            else:
                start_epoch = 0
            if 'iter_count' in checkpoint.keys():
                iter_count  = checkpoint['iter_count'] + 1 # already + 1 when saving
            else:
                iter_count = 0
        else:
            if args.local_rank == 0:
                print('Fresh start...')
            start_epoch = 0
            iter_count = 0
        for epoch in range(start_epoch, p['epochs']):
            train_sampler.set_epoch(epoch)
            if args.local_rank == 0:
                print('Epoch %d/%d' %(epoch+1, p['epochs']))
                print('-'*10)

            end_signal, iter_count = train_phase(p, args, train_dataloader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer_train, tb_writer_test, iter_count, local_rank, nprocs)

            if end_signal:
                break

    # running eval
    if args.local_rank == 0:
        if args.run_mode == 'infer':
            assert(args.trained_model != None)
            checkpoint_path = args.trained_model
            if args.local_rank == 0:
                print('Use checkpoint {}'.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')


            model.load_state_dict(checkpoint['model'], strict=True)
            if 'repara' in p.backbone:
                model.module.backbone.reparameter()
            eval_epoch=0
            SAVE = 0
            if SAVE:
                eval_test = vis_phase(p, test_dataloader, model, criterion, eval_epoch)
            else:
                eval_test = test_phase(p, test_dataloader, model, criterion, eval_epoch)

            print('Infer test restuls:')
            print(eval_test)

        end_time = time.time()
        run_time = (end_time-start_time) / 3600
        print('Total running time: {} h.'.format(run_time))


if __name__ == "__main__":
    # IMPORTANT VARIABLES
    DEBUG_FLAG = True # When True, test the evaluation code when started
    args.nprocs = torch.cuda.device_count()
    print('nprocs is',args.nprocs)
    assert args.run_mode in ['train', 'infer']
    netport = str(random.randint(1,65535))
    print('netport',netport)
    args.netport = netport
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
    # args.nprocs = 1
    # main_worker(0, nprocs=1, args=args)
    # main()