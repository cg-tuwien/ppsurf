import os
import typing
from abc import ABC, abstractmethod

import torch.utils.data as data
import numpy as np

from pytorch_lightning import LightningDataModule

import source.base
import source.base.math
from source.base.proximity import make_kdtree


# Adapted from POCO: https://github.com/valeoai/POCO
# which is published under Apache 2.0: https://github.com/valeoai/POCO/blob/main/LICENSE


def in_file_is_dataset(in_file: str):
    return os.path.splitext(in_file)[1].lower() == '.txt'


def get_dataset_dir(in_file: str):
    dataset_dir = os.path.dirname(in_file)
    return dataset_dir


def get_dataset_name(in_file: str):
    dataset_dir = get_dataset_dir(in_file)
    dataset_name = os.path.basename(dataset_dir)
    return dataset_name


def get_meshes_dir(in_file: str):
    dataset_dir = get_dataset_dir(in_file)
    meshes_dir = os.path.join(dataset_dir, '03_meshes')
    return meshes_dir


def get_pc_dir(in_file: str):
    dataset_dir = get_dataset_dir(in_file)
    pc_dir = os.path.join(dataset_dir, '04_pts_vis')
    return pc_dir


def get_pc_file(in_file, shape_name):
    if in_file_is_dataset(in_file):
        dataset_dir = get_dataset_dir(in_file)
        pc_file = os.path.join(dataset_dir, '04_pts_vis', shape_name + '.xyz.ply')
        return pc_file
    else:
        return in_file


def get_training_data_dir(in_file: str):
    dataset_dir = get_dataset_dir(in_file)
    query_pts_dir = os.path.join(dataset_dir, '05_query_pts')
    query_dist_dir = os.path.join(dataset_dir, '05_query_dist')
    return query_pts_dir, query_dist_dir


def get_set_files(in_file: str):
    if in_file_is_dataset(in_file):
        train_set = os.path.join(os.path.dirname(in_file), 'trainset.txt')
        val_set = os.path.join(os.path.dirname(in_file), 'valset.txt')
        test_set = os.path.join(os.path.dirname(in_file), 'testset.txt')
    else:
        train_set = in_file
        val_set = in_file
        test_set = in_file
    return train_set, val_set, test_set


def get_results_dir(out_dir: str, name: str, in_file: str):
    dataset_name = get_dataset_name(in_file)
    model_results_rec_dir = os.path.join(out_dir, name, dataset_name)
    return model_results_rec_dir


def read_shape_list(shape_list_file: str):
    with open(shape_list_file) as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))
    return shape_names


class OccupancyDataModule(LightningDataModule, ABC):

    def __init__(self, use_ddp, workers, in_file, patches_per_shape: typing.Optional[int],
                 do_data_augmentation: bool, batch_size: int):
        super(OccupancyDataModule, self).__init__()
        self.use_ddp = use_ddp
        self.workers = workers
        self.in_file = in_file
        self.trainset, self.valset, self.testset = get_set_files(in_file)
        self.patches_per_shape = patches_per_shape
        self.do_data_augmentation = do_data_augmentation
        self.batch_size = batch_size

    @staticmethod
    def seed_train_worker(worker_id):
        import random
        import torch
        worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @abstractmethod
    def make_dataset(
            self, in_file: typing.Union[str, list], reconstruction: bool, patches_per_shape: typing.Optional[int],
            do_data_augmentation: bool):
        pass

    def make_datasampler(self, dataset, shuffle=False):
        from torch.cuda import device_count
        if bool(self.use_ddp) and device_count() > 1:
            from torch.utils.data.distributed import DistributedSampler
            datasampler = DistributedSampler(
                dataset, num_replicas=None, rank=None,
                shuffle=shuffle, seed=0, drop_last=False)
        else:
            datasampler = None
        return datasampler

    def make_dataloader(self, dataset, data_sampler, batch_size: int, shuffle: bool = False):

        dataloader = data.DataLoader(
            dataset,
            sampler=data_sampler,
            batch_size=batch_size,
            num_workers=int(self.workers),
            persistent_workers=True if int(self.workers) > 0 else False,
            pin_memory=True,
            worker_init_fn=OccupancyDataModule.seed_train_worker,
            shuffle=shuffle)
        return dataloader

    def train_dataloader(self):
        dataset = self.make_dataset(in_file=self.trainset, reconstruction=False,
                                    patches_per_shape=self.patches_per_shape,
                                    do_data_augmentation=self.do_data_augmentation)
        data_sampler = self.make_datasampler(dataset=dataset, shuffle=True)
        dataloader = self.make_dataloader(dataset=dataset, data_sampler=data_sampler,
                                          batch_size=self.batch_size, shuffle=data_sampler is None)
        return dataloader

    def val_dataloader(self):
        dataset = self.make_dataset(in_file=self.valset, reconstruction=False,
                                    patches_per_shape=self.patches_per_shape, do_data_augmentation=False)
        data_sampler = self.make_datasampler(dataset=dataset, shuffle=False)
        dataloader = self.make_dataloader(dataset=dataset, data_sampler=data_sampler,
                                          batch_size=self.batch_size)
        return dataloader

    def test_dataloader(self):
        batch_size = 1
        dataset = self.make_dataset(in_file=self.testset, reconstruction=False,
                                    patches_per_shape=None, do_data_augmentation=False)
        data_sampler = None
        dataloader = self.make_dataloader(dataset=dataset, data_sampler=data_sampler,
                                          batch_size=batch_size)
        return dataloader

    def predict_dataloader(self):
        batch_size = 1
        dataset = self.make_dataset(in_file=self.testset, reconstruction=True,
                                    patches_per_shape=None, do_data_augmentation=False)
        data_sampler = None
        dataloader = self.make_dataloader(dataset=dataset, data_sampler=data_sampler,
                                          batch_size=batch_size)
        return dataloader

    @staticmethod
    def load_pts(pts_file: str):
        # Supported file formats are:
        # - PLY, STL, OBJ and other mesh files loaded by [trimesh](https://github.com/mikedh/trimesh).
        # - XYZ as whitespace-separated text file, read by [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html).
        # Load first 3 columns as XYZ coordinates. All other columns will be ignored.
        # - NPY and NPZ, read by [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.load.html).
        # NPZ assumes default key='arr_0'. All columns after the first 3 columns will be ignored.
        # - LAS and LAZ (version 1.0-1.4), COPC and CRS loaded by [Laspy](https://github.com/laspy/laspy).
        # You may want to sub-sample large point clouds to ~250k points to avoid speed and memory issues.
        # For detailed reconstruction, you'll need to extract parts of large point clouds.

        import os
        
        file_name, file_ext = os.path.splitext(pts_file)
        file_ext = file_ext.lower()
        if file_ext == '.npy':
            pts = np.load(pts_file)
        elif file_ext == '.npy':
            arrs = np.load(pts_file)
            pts = arrs['arr_0']
        elif file_ext == '.xyz':
            from source.base.point_cloud import load_xyz
            pts = load_xyz(pts_file)
        elif file_ext in ['.stl', '.ply', '.obj', 'gltf', '.glb', '.dae', '.off', '.ctm', '.3dxml']:
            import trimesh
            trimesh_obj: typing.Union[trimesh.Scene, trimesh.Trimesh] = trimesh.load_mesh(file_obj=pts_file)
            if isinstance(trimesh_obj, trimesh.Scene):
                mesh: trimesh.Trimesh = trimesh_obj.geometry.items()[0]
            elif isinstance(trimesh_obj, trimesh.Trimesh):
                mesh: trimesh.Trimesh = trimesh_obj
            elif isinstance(trimesh_obj, trimesh.PointCloud):
                mesh: trimesh.Trimesh = trimesh_obj
            else:
                raise ValueError('Unknown trimesh object type: {}'.format(type(trimesh_obj)))
            pts = np.array(mesh.vertices)
        elif file_ext in ['.las', '.laz', '.copc', '.crs']:
            import laspy
            las = laspy.read(pts_file)
            pts = las.xyz
        else:
            raise ValueError('Unknown point cloud type: {}'.format(pts_file))
        return pts
    
    @staticmethod
    def pre_process_pts(pts: np.ndarray):
        if pts.shape[1] > 3:
            normals = source.base.math.normalize_vectors(pts[:, 3:6])
            pts = pts[:, 0:3]
        else:
            normals = np.zeros_like(pts)
        return pts, normals
        
    @staticmethod
    def load_shape_data_pc(in_file, padding_factor, shape_name: str, normalize=False, return_kdtree=True):
        from source.base import container

        pts_file = get_pc_file(in_file, shape_name)
        pts_np = OccupancyDataModule.load_pts(pts_file=pts_file)
        pts_np, normals_np = OccupancyDataModule.pre_process_pts(pts=pts_np)

        if normalize:
            bb_center, scale = source.base.math.get_points_normalization_info(
                pts=pts_np, padding_factor=padding_factor)
            pts_np = source.base.math.normalize_points_with_info(pts=pts_np, bb_center=bb_center, scale=scale)

        # convert only after normalization
        if pts_np.dtype != np.float32:
            pts_np = pts_np.astype(np.float32)

        # debug output
        from source.base.point_cloud import write_ply
        write_ply('debug/pts_ms.ply', pts_np, normals_np)

        shape_data = {'pts_ms': pts_np, 'normals_ms': normals_np, 'pc_file_in': pts_file}
        if return_kdtree:
            kdtree = make_kdtree(pts_np)
            shape_data['kdtree'] = kdtree

        return shape_data
