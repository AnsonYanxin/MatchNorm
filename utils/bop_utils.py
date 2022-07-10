# Copyright (c) Zheng Dang (zheng.dang@epfl.ch)
# Please cite the following paper if you use any part of the code.
# [-] Zheng Dang, Lizhou Wang, Yu Guo, Mathieu Salzmann, Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World, ECCV2022

import os, sys
import trimesh
import numpy as np
from copy import deepcopy
sys.path.append('/home/dz/2021_code_nas/XEPose/bop_toolkit')
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout

def get_bop_dataset(datasets, partition, split_type):
    data_list = []
    for dataset in datasets:
        seq_list = get_bop_dataset_single(dataset, partition, split_type)
        data_list.extend(seq_list)
    return data_list

def get_bop_dataset_single(dataset, partition, split_type):
    targets_filename = None if partition == 'train' else 'test_targets_bop19.json'
    if dataset == 'lm':
        split_type = 'pbr' if partition == 'train' else None
        p = {
        'dataset': dataset,
        'dataset_split': partition,
        'dataset_split_type': split_type,
        'targets_filename': targets_filename,
        # 'scene_ids': list(np.arange(1, 16)),
        'datasets_path': config.datasets_path,
        }
    elif dataset == 'lmo':
        p = {
        'dataset': dataset,
        'dataset_split': partition,
        'dataset_split_type': split_type,
        'targets_filename': targets_filename,
        # 'scene_ids': [2],
        'datasets_path': config.datasets_path,
        }
    elif dataset == 'ycbv':
        split_type = 'real' if partition == 'train' else None
        p = {
        'dataset': dataset,
        'dataset_split': partition,
        'dataset_split_type': split_type,
        'targets_filename': targets_filename,
        # 'scene_ids': list(np.arange(48, 60)),
        'datasets_path': config.datasets_path,
        }
    elif dataset == 'icbin':
        p = {
        'dataset': dataset,
        'dataset_split': partition,
        'dataset_split_type': split_type,
        'targets_filename': targets_filename,
        # 'scene_ids': [1,2,3],
        'datasets_path': config.datasets_path,
        }
    elif dataset == 'tudl':
        split_type = 'real' if partition == 'train' else None
        # parameters for bop_challenge
        p = {
        'dataset': dataset,
        'dataset_split': partition,
        'dataset_split_type': split_type,
        'targets_filename': targets_filename,
        # 'scene_ids': [1,2,3],
        'datasets_path': config.datasets_path,
        }
        # parameters for cvpr2022 paper
        # p = {
        # 'dataset': dataset,
        # 'dataset_split': partition,
        # 'dataset_split_type': split_type,
        # 'targets_filename': None,
        # 'scene_ids': [1,2,3], # tudl testing sequence
        # 'datasets_path': config.datasets_path,
        # }
    elif dataset == 'hb':
        p = {
        'dataset': dataset,
        'dataset_split': 'test',
        'dataset_split_type': split_type,
        'targets_filename': targets_filename,
        # 'scene_ids': [3, 5, 13],
        'datasets_path': config.datasets_path,
        }
        # parameters for cvpr2022 paper
        # p = {
        # 'dataset': dataset,
        # # 'dataset_split': 'train',
        # 'dataset_split': 'val',
        # # 'dataset_split_type': 'kinect',
        # # 'dataset_split_type': 'primesense',
        # 'dataset_split_type': split_type,
        # 'targets_filename': None,
        # 'scene_ids': list(range(1, 14)), # hb all testing sequence
        # 'datasets_path': config.datasets_path,
        # }
    elif dataset == 'tless':
        p = {
        'dataset': dataset,
        'dataset_split': 'test',
        'dataset_split_type': split_type,
        'targets_filename': targets_filename,
        # 'scene_ids': list(np.arange(1, 21)),
        'datasets_path': config.datasets_path,
        }
    elif dataset == 'itodd':
        p = {
        'dataset': dataset,
        'dataset_split': 'test',
        'dataset_split_type': split_type,
        'targets_filename': targets_filename,
        # 'scene_ids': [1],
        'datasets_path': config.datasets_path,
        }
    elif dataset == 'xepose':
        p = {
        'dataset': dataset,
        'dataset_split': partition,
        'dataset_split_type': split_type,
        'targets_filename': None,
        'scene_ids': [203, 204], # xepose testing sequence
        # 'scene_ids': list(range(48, 60)), # ycbv testing sequence
        'datasets_path': config.datasets_path,
        }
    else:
        raise('not implement yet.')
    # Load dataset parameters.
    dp_split = dataset_params.get_split_params(p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])
    mp_split = dataset_params.get_model_params(p['datasets_path'], p['dataset'])
    models_info = inout.load_json(mp_split['models_info_path'], keys_to_int=True)

    from collections import defaultdict
    if p['targets_filename'] is not None:
        targets = inout.load_json(os.path.join(dp_split['base_path'], p['targets_filename']))
        scene_file = defaultdict(list)
        data_list = []
        for target in targets:
            scene_file[target['scene_id']].append({'im_id': target['im_id'], 'obj_id': target['obj_id']})
        for scene_id in scene_file.keys():
            scene_camera = inout.load_scene_camera(dp_split['scene_camera_tpath'].format(scene_id=scene_id))
            scene_gt = inout.load_scene_gt(dp_split['scene_gt_tpath'].format(scene_id=scene_id))
            scene_gt_info = inout.load_json(dp_split['scene_gt_info_tpath'].format(scene_id=scene_id), keys_to_int=True)
            for item in scene_file[scene_id]:
                im_id, obj_id = item['im_id'], item['obj_id']
                for gt_id, (gt, gt_info) in enumerate(zip(scene_gt[im_id], scene_gt_info[im_id])):
                    if gt['obj_id'] == obj_id:
                        A = {
                            'scene_id': scene_id,
                            'im_id': im_id,
                            'scene_gt': gt,
                            'scene_gt_info': gt_info,
                            'scene_camera': scene_camera[im_id],
                            'gt_id': gt_id,
                            'diameter': models_info[gt['obj_id']]['diameter'],
                            'dp_split': dp_split,
                            'mp_split': mp_split,
                            }
                        data_list.append(A)
    else:
        data_list = []
        scene_ids = dataset_params.get_present_scene_ids(dp_split)
        for scene_id in scene_ids:
            scene_camera = inout.load_scene_camera(dp_split['scene_camera_tpath'].format(scene_id=scene_id))
            scene_gt = inout.load_scene_gt(dp_split['scene_gt_tpath'].format(scene_id=scene_id))
            scene_gt_info = inout.load_json(dp_split['scene_gt_info_tpath'].format(scene_id=scene_id), keys_to_int=True)
            seq_list = []
            im_ids = sorted(scene_gt.keys())
            for im_id in im_ids:
                for gt_id, (gt, gt_info) in enumerate(zip(scene_gt[im_id], scene_gt_info[im_id])):
                    if partition == 'train' and gt_info['px_count_valid'] < 2000: continue
                    if partition == 'train' and gt_info['visib_fract'] < .5: continue
                    A = {
                        'scene_id': scene_id,
                        'im_id': im_id,
                        'scene_gt': gt,
                        'scene_gt_info': gt_info,
                        'scene_camera': scene_camera[im_id],
                        'gt_id': gt_id,
                        'diameter': models_info[gt['obj_id']]['diameter'],
                        'dp_split': dp_split,
                        'mp_split': mp_split,
                        }
                    seq_list.append(A)
            data_list.extend(seq_list)
    return data_list

def get_bop_models(dataset_list, repeat=None):
    data_list = []
    for dataset in dataset_list:
        p = {
        'dataset': dataset,
        'datasets_path': config.datasets_path,
        }
        # Load dataset parameters.
        mp_split = dataset_params.get_model_params(p['datasets_path'], p['dataset'])
        models_info = inout.load_json(mp_split['models_info_path'], keys_to_int=True)
        for obj_id in mp_split['obj_ids']:
            # tri_mesh = o3d.io.read_triangle_mesh((mp_split['model_tpath'].format(obj_id=obj_id)))
            tri_mesh = trimesh.load((mp_split['model_tpath'].format(obj_id=obj_id)))
            A = {
                'mesh': tri_mesh,
                'diameter': models_info[obj_id]['diameter'],
            }
            data_list.append(A)

    if repeat is not None:
        item = deepcopy(data_list)
        for _ in range(repeat): data_list.extend(item)
    return data_list