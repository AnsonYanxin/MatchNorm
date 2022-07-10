# Copyright (c) Zheng Dang (zheng.dang@epfl.ch)
# Please cite the following paper if you use any part of the code.
# [-] Zheng Dang, Lizhou Wang, Yu Guo, Mathieu Salzmann, Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World, ECCV2022

import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform.rotation import Rotation
import sys;sys.path.append('/home/dz/2021_code_nas/XEPose/bop_toolkit')
from bop_toolkit_lib import inout
from utils.utils import build_bijective_label, verts2pt3d, voxelization
from utils.bop_utils import get_bop_dataset
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

class BOP_Base(Dataset):
    def __init__(self, dataset, partition, split_type, format=None, pose_type=None, subset=None):
        super().__init__()
        self.dataset = dataset
        self.partition = partition
        self.format = format
        self.pose_type = pose_type
        self.subset = subset
        self.sampling_type = 'from_mesh'
        self.rot_aug = True # augmentation in the training process

        if format == 'bopdataset' and pose_type == 'read_file':
            self.data = get_bop_dataset([dataset], partition, split_type)
        else:
            raise('not implement yet.')
        print(dataset, partition, split_type, len(self.data))

        R_max = np.asarray([ np.pi,  np.pi/2,  np.pi])
        R_min = np.asarray([-np.pi, -np.pi/2, -np.pi])
        self.R_list = (R_max, R_min)

        self.t_max = np.asarray([ 30,  30, 1100])
        self.t_min = np.asarray([-30, -30, 1000])
    
    @staticmethod
    def crop_mask(mask):
        mask_crop = cv2.erode(mask, kernel=np.ones((2, 2), np.uint8), iterations=2)
        return mask_crop

    def prepare_mesh(self, index):
        ''' 
        using index load the model then convert to mesh.'''
        if self.format == 'bopdataset' and self.pose_type == 'read_file':
            scene_gt = self.data[index]['scene_gt']
            mp_split = self.data[index]['mp_split']
            triangle_mash = o3d.io.read_triangle_mesh((mp_split['model_tpath'].format(obj_id=scene_gt['obj_id'])))
            verts = torch.from_numpy(np.asarray(triangle_mash.vertices))
            faces = torch.from_numpy(np.asarray(triangle_mash.triangles))
        else:
            raise('not implement yet.')
        verts = verts.float()
        faces = faces.int()

        models_scale = torch.tensor([0.001]).float() # mm to meter
        obj_size = (max(verts.max(), np.abs(verts.min())))
        verts *= models_scale

        if self.sampling_type == 'from_mesh':
            # 1. uniformly sampling from the surface of the mesh model.
            ori_mesh = Meshes(verts=[verts], faces=[faces])
            pts1 = sample_points_from_meshes(ori_mesh, self.mesh_sampling_pts).squeeze(0)
            pts1 = pts1.numpy()

        if self.pose_type == 'read_file':
            cam_R_m2c = torch.from_numpy(scene_gt['cam_R_m2c']).float() # pose of the object.
            cam_t_m2c = torch.from_numpy(scene_gt['cam_t_m2c']).float() * models_scale
            if self.partition == 'train' and self.rot_aug is True:
                R_aug = torch.from_numpy(Rotation.from_euler('zyx', np.random.uniform([0., 0., 0.], [np.pi/2, np.pi/2, np.pi/2])).as_matrix().astype('float32'))
                pts1 = (R_aug @ (verts.T)).T
                pts1 = pts1.numpy()
                cam_R_m2c = cam_R_m2c @ R_aug.T
        else:
            raise('not implement yet.')

        # transfrom the verts for rendering the target depth map.
        verts = cam_R_m2c @ verts.T + cam_t_m2c
        # verts to pytorch3d coordinate
        verts = verts2pt3d(verts)
        # Create a Meshes object. Here we have only one mesh in the batch.
        verts = verts.T.float()
        diameter = self.data[index]['diameter'] if self.format == 'bopdataset' else 1.
        model_info = {
            'scale': models_scale,
            'diameter': diameter,
            'obj_size': obj_size, # only for bpnet2 and bop_datasets
        }
        return verts, faces, cam_R_m2c, cam_t_m2c, pts1, model_info


class BOP_Dataset(BOP_Base):
    def __init__(self, args, partition=None, split_type=None, dataset='modelnet40', format='pytorch3d', pose_type='random', mask_type='render', subset=None):
        self.args = args
        super().__init__(dataset, partition, split_type, format, pose_type, subset)

        self.mesh_sampling_pts = 1024*4
        self.src_points = 1024
        self.tgt_points = 768
        self.dataset = dataset
        self.mask_type = mask_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if self.partition != 'train':
            torch.manual_seed(index)
            torch.cuda.manual_seed_all(index)
            np.random.seed(index)
        
        if self.mask_type == 'gt':
            _, _, cam_R_m2c, cam_t_m2c, pts1, model_info = self.prepare_mesh(index)
            models_scale = model_info['scale']
            scene_id = self.data[index]['scene_id']
            im_id = self.data[index]['im_id']
            gt_id = self.data[index]['gt_id']
            scene_camera = self.data[index]['scene_camera']
            dp_split = self.data[index]['dp_split']
            info_dict = {
                'scene_id': scene_id,
                'im_id': im_id,
                'obj_id': self.data[index]['scene_gt']['obj_id'],
            }

            mask = inout.load_im(dp_split['mask_visib_tpath'].format(scene_id=scene_id, im_id=im_id, gt_id=gt_id)) 
            depth = inout.load_depth(dp_split['depth_tpath'].format(scene_id=scene_id, im_id=im_id))
            depth *= scene_camera['depth_scale']  # Convert to [mm].
            mask = np.array(mask) / 255
            if self.dataset in ['lm']: mask = self.crop_mask(mask)
            depth = np.array(depth) * mask
            # refine process for real dataset.
            tgt = self.refine_mask_median(mask, depth, scene_camera, th=model_info['obj_size'].numpy())

            pts2 = tgt * models_scale.numpy()
            if self.partition == 'train' and self.rot_aug is True:
                R_aug = torch.from_numpy(Rotation.from_euler('zyx', np.random.uniform([0., 0., 0.], [np.pi/2, np.pi/2, np.pi/2])).as_matrix().astype('float32'))
                pts2 = (R_aug @ (pts2.T)).T.numpy()
                cam_R_m2c = R_aug @ cam_R_m2c
                cam_t_m2c = R_aug @ cam_t_m2c
        else:
            raise('mask_type not implement yet')

        # voxelize the point cloud.
        source, target = voxelization(pts1, pts2, 0.02, self.src_points, self.tgt_points)
        R_gt, t_gt = cam_R_m2c.numpy(), cam_t_m2c.numpy()

        label = build_bijective_label(source, target, R_gt, t_gt, self.args.th, self.args.val, self.args.bi_layer)
        info_dict['model_scale'] = models_scale
        info_dict['diameter'] = model_info['diameter']
        info_dict['inlier_total'] = label[:-1, :-1].sum()
        return source.T.astype('float32'), target.T.astype('float32'), R_gt.astype('float32'), t_gt.astype('float32'), label, info_dict

    def refine_mask_median(self, mask, depth, scene_camera, th=200):
        '''
        Get the median depth value in the object's depth map. Then using it to
        get rid of the outlier which far from the target object's surface.
        '''
        x, y = np.where(mask > 0)
        z = depth[x, y]
        C_inv = np.linalg.inv(scene_camera['cam_K'])
        ones = np.ones_like(x)
        xy1 = np.stack([y, x, ones], axis=0)
        xy1_cam = C_inv @ xy1
        tgt = (xy1_cam * z[None]).T

        tgt_median = np.median(tgt[:, 2])
        mask = np.abs(tgt[:, 2] - tgt_median) < th 
        return tgt[mask]
