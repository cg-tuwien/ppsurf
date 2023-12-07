import os
import os.path
import typing

import torch.utils.data as torch_data
import numpy as np
import torch
import trimesh
from overrides import EnforceOverrides, overrides

from source.occupancy_data_module import OccupancyDataModule, get_training_data_dir
from source.base.container import dict_np_to_torch


# Adapted from POCO: https://github.com/valeoai/POCO
# which is published under Apache 2.0: https://github.com/valeoai/POCO/blob/main/LICENSE


class PocoDataModule(OccupancyDataModule):

    def __init__(self, in_file, workers, use_ddp, padding_factor, seed, manifold_points,
                 patches_per_shape: typing.Optional[int], do_data_augmentation: bool, batch_size: int):
        super(PocoDataModule, self).__init__(
            use_ddp=use_ddp, workers=workers, in_file=in_file, patches_per_shape=patches_per_shape,
            do_data_augmentation=do_data_augmentation, batch_size=batch_size)
        self.in_file = in_file
        self.padding_factor = padding_factor
        self.seed = seed
        self.manifold_points = manifold_points
        self.patches_per_shape = patches_per_shape
        self.do_data_augmentation = do_data_augmentation

    def make_dataset(
            self, in_file: typing.Union[str, list], reconstruction: bool, patches_per_shape: typing.Optional[int],
            do_data_augmentation: bool):
        if reconstruction:
            dataset = PocoReconstructionDataset(
                in_file=in_file,
                padding_factor=self.padding_factor,
                seed=self.seed,
                use_ddp=self.use_ddp,
            )
        else:
            dataset = PocoDataset(
                in_file=in_file,
                padding_factor=self.padding_factor,
                seed=self.seed,
                patches_per_shape=self.patches_per_shape,
                do_data_augmentation=do_data_augmentation,
                use_ddp=self.use_ddp,
                manifold_points=self.manifold_points,
            )
        return dataset


def sampling_quantized(pts_batch, ratio=None, n_support=None, support_points=None, support_points_ids=None):
    import math
    import torch
    from torch_geometric.transforms import RandomRotate
    from torch_geometric.data import Data
    from torch_geometric.nn import voxel_grid
    from torch_geometric.nn.pool.consecutive import consecutive_cluster
    from source.base.nn import batch_gather

    if support_points is not None:
        return support_points, support_points_ids

    assert ((ratio is None) != (n_support is None))

    if ratio is not None:
        support_point_number = max(1, int(pts_batch.shape[2] * ratio))
    else:
        support_point_number = n_support

    if support_point_number == pts_batch.shape[2]:
        support_points_ids = torch.arange(pts_batch.shape[2], dtype=torch.long, device=pts_batch.device)
        support_points_ids = support_points_ids.unsqueeze(0).expand(pts_batch.shape[0], pts_batch.shape[2])
        return pts_batch, support_points_ids
    elif 0 < support_point_number < pts_batch.shape[2]:

        # voxel_size
        maxi, _ = torch.max(pts_batch, dim=2)
        mini, _ = torch.min(pts_batch, dim=2)
        vox_size = (maxi - mini).norm(2, dim=1) / math.sqrt(support_point_number)

        rot_x = RandomRotate(180, axis=0)
        rot_y = RandomRotate(180, axis=1)
        rot_z = RandomRotate(180, axis=2)

        support_points_ids = []
        for i in range(pts_batch.shape[0]):
            pts = pts_batch[i].clone().transpose(0, 1)
            ids = torch.arange(pts.shape[0], device=pts.device)
            sampled_count = 0
            sampled = []
            vox = vox_size[i]
            while True:
                #data = Data(pos=pts)
                # TODO: optimize to one call to linear transformation
                pts_rot = rot_z(rot_y(rot_x(Data(pos=pts)))).pos.to(pts.dtype)

                c = voxel_grid(pts_rot, batch=torch.zeros(pts_rot.shape[0], device=pts.device, dtype=pts.dtype), size=vox)
                _, perm = consecutive_cluster(c)

                if sampled_count + perm.shape[0] < support_point_number:
                    sampled.append(ids[perm])
                    sampled_count += perm.shape[0]

                    tmp = torch.ones_like(ids)
                    tmp[perm] = 0
                    tmp = (tmp > 0)
                    pts = pts[tmp]
                    ids = ids[tmp]
                    vox = vox / 2
                    # pts = pts[perm]
                    # ids = ids[perm]
                else:
                    n_to_select = support_point_number - sampled_count
                    perm = perm[torch.randperm(perm.shape[0])[:n_to_select]]
                    sampled.append(ids[perm])
                    break
            sampled = torch.cat(sampled)
            support_points_ids.append(sampled)

        support_points_ids = torch.stack(support_points_ids, dim=0)
        support_points_ids = support_points_ids.to(pts_batch.device)
        support_points = batch_gather(pts_batch, dim=2, index=support_points_ids)
        return support_points, support_points_ids
    else:
        raise ValueError(f'Search Quantized - ratio value error {ratio} should be in ]0,1]')


def get_fkaconv_ids(data: typing.Dict[str, torch.Tensor], segmentation: bool = True) \
        -> typing.Dict[str, torch.Tensor]:
    from source.poco_utils import knn

    pts = data['pts'].clone()

    add_batch_dimension = False
    if len(pts.shape) == 2:
        pts = pts.unsqueeze(0)
        add_batch_dimension = True

    support1, _ = sampling_quantized(pts, 0.25)
    support2, _ = sampling_quantized(support1, 0.25)
    support3, _ = sampling_quantized(support2, 0.25)
    support4, _ = sampling_quantized(support3, 0.25)

    # compute the ids
    ret_data = {}
    ids00 = knn(pts, pts, 16)
    ids01 = knn(pts, support1, 16)
    ids11 = knn(support1, support1, 16)
    ids12 = knn(support1, support2, 16)
    ids22 = knn(support2, support2, 16)
    ids23 = knn(support2, support3, 16)
    ids33 = knn(support3, support3, 16)
    ids34 = knn(support3, support4, 16)
    ids44 = knn(support4, support4, 16)
    if segmentation:
        ids43 = knn(support4, support3, 1)
        ids32 = knn(support3, support2, 1)
        ids21 = knn(support2, support1, 1)
        ids10 = knn(support1, pts, 1)
        if add_batch_dimension:
            ids43 = ids43.squeeze(0)
            ids32 = ids32.squeeze(0)
            ids21 = ids21.squeeze(0)
            ids10 = ids10.squeeze(0)

        ret_data['ids43'] = ids43
        ret_data['ids32'] = ids32
        ret_data['ids21'] = ids21
        ret_data['ids10'] = ids10

    if add_batch_dimension:
        support1 = support1.squeeze(0)
        support2 = support2.squeeze(0)
        support3 = support3.squeeze(0)
        support4 = support4.squeeze(0)
        ids00 = ids00.squeeze(0)
        ids01 = ids01.squeeze(0)
        ids11 = ids11.squeeze(0)
        ids12 = ids12.squeeze(0)
        ids22 = ids22.squeeze(0)
        ids23 = ids23.squeeze(0)
        ids33 = ids33.squeeze(0)
        ids34 = ids34.squeeze(0)
        ids44 = ids44.squeeze(0)

    ret_data['support1'] = support1
    ret_data['support2'] = support2
    ret_data['support3'] = support3
    ret_data['support4'] = support4

    ret_data['ids00'] = ids00
    ret_data['ids01'] = ids01
    ret_data['ids11'] = ids11
    ret_data['ids12'] = ids12
    ret_data['ids22'] = ids22
    ret_data['ids23'] = ids23
    ret_data['ids33'] = ids33
    ret_data['ids34'] = ids34
    ret_data['ids44'] = ids44
    return ret_data


def get_proj_ids(data: typing.Dict[str, torch.Tensor], k: int) -> typing.Dict[str, torch.Tensor]:
    from source.poco_utils import knn

    pts = data['pts']
    pts_query = data['pts_query']

    add_batch_dimension_pos = False
    if len(pts.shape) == 2:
        pts = pts.unsqueeze(0)
        add_batch_dimension_pos = True

    add_batch_dimension_non_manifold = False
    if len(pts_query.shape) == 2:
        pts_query = pts_query.unsqueeze(0)
        add_batch_dimension_non_manifold = True

    if pts.shape[1] != 3:
        pts = pts.transpose(1, 2)

    if pts_query.shape[1] != 3:
        pts_query = pts_query.transpose(1, 2)

    indices = knn(pts, pts_query, k, -1)

    if add_batch_dimension_non_manifold or add_batch_dimension_pos:
        indices = indices.squeeze(0)

    ret_data = {'proj_ids': indices}
    return ret_data


def get_data_poco(batch_data: dict):
    import torch

    fkaconv_data = {
        'pts': torch.transpose(batch_data['pts_ms'], -1, -2),
        'pts_query': torch.transpose(batch_data['pts_query_ms'], -1, -2),
    }

    if 'imp_surf_dist_ms' in batch_data.keys():
        occ_sign = torch.sign(batch_data['imp_surf_dist_ms'])
        occ = torch.zeros_like(occ_sign, dtype=torch.int64)
        occ[occ_sign > 0.0] = 1
        fkaconv_data['occ'] = occ
    else:
        fkaconv_data['occ'] = torch.zeros(fkaconv_data['pts_query'].shape[:1])

    with torch.no_grad():
        net_data = get_fkaconv_ids(fkaconv_data)
        proj_data = get_proj_ids(fkaconv_data, k=64)  # TODO: put k in param
        net_data['proj_ids'] = proj_data['proj_ids']

    # need points also for poco ids
    for k in fkaconv_data.keys():
        batch_data[k] = fkaconv_data[k]
    for k in net_data.keys():
        batch_data[k] = net_data[k]

    return batch_data


class PocoDataset(torch_data.Dataset, EnforceOverrides):

    def __init__(self, in_file: str, padding_factor: float, seed, use_ddp: bool, manifold_points: typing.Optional[int],
                 patches_per_shape: typing.Optional[int], do_data_augmentation=True):

        super(PocoDataset, self).__init__()

        self.in_file = in_file
        self.seed = seed
        self.patches_per_shape = patches_per_shape
        self.do_data_augmentation = do_data_augmentation
        self.padding_factor = padding_factor
        self.use_ddp = use_ddp
        self.manifold_points = manifold_points

        # initialize rng for picking points in the local subsample of a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2 ** 32 - 1, 1)[0]

        from torch.cuda import device_count
        if bool(self.use_ddp) and device_count() > 1:
            import torch.distributed as dist
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()
            self.seed += rank
        self.rng = np.random.RandomState(self.seed)

        # get all shape names in the dataset
        if isinstance(self.in_file, str):
            # assume .txt files contain a list of shapes
            if os.path.splitext(self.in_file)[1].lower() == '.txt':
                self.shape_names = []
                with open(os.path.join(in_file)) as f:
                    self.shape_names = f.readlines()
                self.shape_names = [x.strip() for x in self.shape_names]
                self.shape_names = list(filter(None, self.shape_names))
            else:  # all other single files are just one shape to be reconstructed
                self.shape_names = [self.in_file]
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.shape_names)

    def augment_shape(self, shape_data: dict, rand_rot: np.ndarray) -> dict:
        import trimesh.transformations as trafo

        def rot_arr(arr, rot):
            return trafo.transform_points(arr, rot).astype(np.float32)

        shape_data['pts_ms'] = rot_arr(shape_data['pts_ms'], rand_rot)
        shape_data['normals_ms'] = rot_arr(shape_data['normals_ms'], rand_rot)
        shape_data['pts_query_ms'] = rot_arr(shape_data['pts_query_ms'], rand_rot)
        return shape_data

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the patch center
    def __getitem__(self, shape_id):
        shape_data, pts_ms_raw = self.load_shape_by_index(shape_id, return_kdtree=False)

        if self.do_data_augmentation:
            # self.rng.seed(42)  # always pick the same points for debugging
            rand_rot = trimesh.transformations.random_rotation_matrix(self.rng.rand(3))
            shape_data = self.augment_shape(shape_data, rand_rot)

        shape_data = dict_np_to_torch(shape_data)
        shape_data = get_data_poco(shape_data)
        return shape_data

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind, return_kdtree=True):
        # assume that datasets are already normalized
        from source.occupancy_data_module import in_file_is_dataset
        normalize = not in_file_is_dataset(self.in_file)

        shape_data = OccupancyDataModule.load_shape_data_pc(
            in_file=self.in_file, padding_factor=self.padding_factor,
            shape_name=self.shape_names[shape_ind], normalize=normalize, return_kdtree=return_kdtree)
        pts_ms_raw = shape_data['pts_ms']

        def sub_sample_point_cloud(pts: np.ndarray, normals: np.ndarray, num_target_pts: int):
            if num_target_pts is None:
                return pts, normals
            replace = True if pts.shape[0] < num_target_pts else False
            choice_ids = self.rng.choice(np.arange(pts.shape[0]), size=num_target_pts, replace=replace)
            return pts[choice_ids], normals[choice_ids]

        pts_sub_sample, normals_sub_sample = sub_sample_point_cloud(
            pts=shape_data['pts_ms'], normals=shape_data['normals_ms'], num_target_pts=self.manifold_points)
        shape_data['pts_ms'] = pts_sub_sample
        shape_data['normals_ms'] = normals_sub_sample

        query_pts_dir, query_dist_dir = get_training_data_dir(self.in_file)
        imp_surf_query_filename = os.path.join(query_pts_dir, self.shape_names[shape_ind] + '.ply.npy')
        imp_surf_dist_filename = os.path.join(query_dist_dir, self.shape_names[shape_ind] + '.ply.npy')

        if os.path.isfile(imp_surf_query_filename):  # if GT data exists
            pts_query_ms = np.load(imp_surf_query_filename)
            if pts_query_ms.dtype != np.float32:
                pts_query_ms = pts_query_ms.astype(np.float32)

            imp_surf_dist_ms = np.load(imp_surf_dist_filename)
            if imp_surf_dist_ms.dtype != np.float32:
                imp_surf_dist_ms = imp_surf_dist_ms.astype(np.float32)
        else:  # if no GT data
            pts_query_ms = np.empty((0, 3), dtype=np.float32)
            imp_surf_dist_ms = np.empty((0, 3), dtype=np.float32)

        # DDP sampler can't handle patches_per_shape, so we do it here
        from torch.cuda import device_count
        if bool(self.use_ddp) and device_count() > 1 and \
                self.patches_per_shape is not None and self.patches_per_shape > 0:
            query_pts_ids = self.rng.choice(np.arange(pts_query_ms.shape[0]), self.patches_per_shape)
            pts_query_ms = pts_query_ms[query_pts_ids]
            imp_surf_dist_ms = imp_surf_dist_ms[query_pts_ids]

        shape_data['pts_query_ms'] = pts_query_ms
        shape_data['imp_surf_dist_ms'] = imp_surf_dist_ms
        shape_data['shape_id'] = shape_ind

        # print('PID={}: loaded shape {}'.format(os.getpid(), shape_id))  # debug multi-processing cache

        return shape_data, pts_ms_raw


class PocoReconstructionDataset(PocoDataset):

    def __init__(self, in_file, padding_factor, seed, use_ddp):
        super(PocoReconstructionDataset, self).__init__(
            in_file=in_file, padding_factor=padding_factor, seed=seed,
            use_ddp=use_ddp, manifold_points=None,
            patches_per_shape=None, do_data_augmentation=False)

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the patch center
    def __getitem__(self, shape_id):
        shape_data, pts_ms_raw = self.load_shape_by_index(shape_id, return_kdtree=False)
        shape_data = dict_np_to_torch(shape_data)
        return shape_data
