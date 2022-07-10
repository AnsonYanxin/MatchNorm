# Copyright (c) Zheng Dang (zheng.dang@epfl.ch)
# Please cite the following paper if you use any part of the code.
# [-] Zheng Dang, Lizhou Wang, Yu Guo, Mathieu Salzmann, Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World, ECCV2022

import os
import torch
from termcolor import colored
from utils.utils import dump_checkpoint, mAP_report, AverageMeterGroup, load_checkpoint, save_best

def train_iteration(args, model, criterion, optimizer, loader_tr, loader_val, comp_tr, comp_val, logger=None):
    ckpt_latest_path = "ckpt/" + args.exp_name + "/ckpt_latest.pth.tar"
    if os.path.exists(ckpt_latest_path):
        model, optimizer, iters_start, best_score = load_checkpoint(model, optimizer, ckpt_latest_path)
        logger.info(f'Load the checkpoint:{ckpt_latest_path}')
    else:
        iters_start = 0 
        best_score = 0
    model.train()
    data_iter = iter(loader_tr)
    for iter_idx in range(iters_start, args.iters):
        try:
            batch = data_iter.__next__()
        except(StopIteration):
            data_iter = iter(loader_tr)
            iter_idx -= 1 # avoid the first batch has error point
            continue
        except(Exception):
            print('Extreme case happen, skip the failure batch.')
            continue
        src, tgt, R_gt, t_gt, score, info_dict = batch
        optimizer.zero_grad()
        pred = model(src.cuda(), tgt.cuda())
        
        loss = criterion(pred, score.cuda())
        loss.backward()
        optimizer.step()

        if iter_idx % args.log_frequency == 0:
            meters = AverageMeterGroup()
            with torch.no_grad():
                metrics = comp_tr(src.cuda(), tgt.cuda(), R_gt, t_gt, pred, info_dict)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            mAP = mAP_report(meters)
            logger.info(colored(f'\nIter [{iter_idx}/{args.iters}] Finished Epoch [{iter_idx // len(loader_tr)}] '+\
                        mAP+\
                        f'\nLoss:{meters.loss.val:.2f} (avg:{meters.loss.avg:.2f})', 'blue' )
                        )

        if iter_idx % args.log_intv == 0 and iter_idx > 5001:
        # if iter_idx % args.log_intv == 0 and iter_idx > 1:
            model.eval()
            meters = AverageMeterGroup()
            with torch.no_grad():
                for step, (src, tgt, R_gt, t_gt, _, info_dict) in enumerate(loader_val):
                    # src, tgt = src.cuda(), tgt.cuda()
                    pred = model(src.cuda(), tgt.cuda())
                    metrics = comp_val(src.cuda(), tgt.cuda(), R_gt, t_gt, pred, info_dict)
                    meters.update(metrics)
                    mAP = mAP_report(meters)
                    if step % args.log_frequency == 0 or step + 1 == len(loader_val):
                        logger.info(colored(f'\nIteration [{iter_idx}/{args.iters}] Val Step [{step}] '+mAP, 'green' ))
            # logger.info(colored(f'\nIteration [{iter_idx}] validation '+mAP, 'green'))
            curr_score = meters.mAP_R5.avg
            if curr_score > best_score:
                best_score = curr_score
                logger.info(f'Save the best model at iter: {iter_idx}')
                save_best(model, "ckpt/" + args.exp_name, logger)
            dump_checkpoint(model, optimizer, iter_idx, best_score, "ckpt/" + args.exp_name, logger=logger)
            model.train()
     
def validate(epoch, model, loader, args, comp_func, logger=None, dataset_name=None):
    model.eval()
    meters = AverageMeterGroup()
    
    # file_name = f'/home/dz/2022_code/bop_res/zheng-bpnet_{dataset_name}-test.csv'
    file_name = f'./bop_res/zheng-bpnet_{dataset_name}-test.csv'
    if os.path.exists(file_name): os.remove(file_name)

    with torch.no_grad():
        for step, (src, tgt, R_gt, t_gt, _, info_dict) in enumerate(loader):
            pred = model(src.cuda(), tgt.cuda())
            metrics = comp_func(src.cuda(), tgt.cuda(), R_gt, t_gt, pred, info_dict, file_name=file_name)
            meters.update(metrics)
            if step % args.log_frequency == 0 or step + 1 == len(loader):
                logger.info(colored(f'\nEpoch [{epoch + 1}/{args.epochs}] Val Step [{step + 1}/{len(loader)}] ' 
                            f'\nmAP_R:[{meters.mAP_R5.avg:.2f}, {meters.mAP_R10.avg:.2f}, {meters.mAP_R15.avg:.2f}, '
                            f'{meters.mAP_R20.avg:.2f}, {meters.mAP_R25.avg:.2f}, {meters.mAP_R30.avg:.2f}] '
                            f'(avg:{meters.mAP_R_mean.avg:.2f})'
                            f'\nmAP_t:[{meters.mAP_t5.avg:.2f}, {meters.mAP_t10.avg:.2f}, {meters.mAP_t15.avg:.2f}, '
                            f'{meters.mAP_t20.avg:.2f}, {meters.mAP_t25.avg:.2f}, {meters.mAP_t30.avg:.2f}] '
                            f'(avg:{meters.mAP_t_mean.avg:.2f})'
                            f'\nADD:[{meters.add.avg:.2f}] ADDS:[{meters.adds.avg:.2f}]'
                            f'\nInlier: pred:[{meters.inlier_pred.avg:.2f}], true:[{meters.inlier_true.avg:.2f}]'
                            , 'green' )
                            )

    logger.info(colored(f'\nEpoch [{epoch + 1}] validation ' 
                f'\nmAP_R:[{meters.mAP_R5.avg:.2f}, {meters.mAP_R10.avg:.2f}, {meters.mAP_R15.avg:.2f}, '
                f'{meters.mAP_R20.avg:.2f}, {meters.mAP_R25.avg:.2f}, {meters.mAP_R30.avg:.2f}] '
                f'(avg:{meters.mAP_R_mean.avg:.2f}) '
                f'\nmAP_t:[{meters.mAP_t5.avg:.2f}, {meters.mAP_t10.avg:.2f}, {meters.mAP_t15.avg:.2f}, '
                f'{meters.mAP_t20.avg:.2f}, {meters.mAP_t25.avg:.2f}, {meters.mAP_t30.avg:.2f}] '
                f'(avg:{meters.mAP_t_mean.avg:.2f}) '
                f'\nADD:[{meters.add.avg:.2f}] ADDS:[{meters.adds.avg:.2f}]'
                f'\nInlier: pred:[{meters.inlier_pred.avg:.2f}], true:[{meters.inlier_true.avg:.2f}]'
                , 'green')
                )