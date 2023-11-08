import typing

import numpy as np
from overrides import overrides

from source.poco_data_loader import PocoDataModule, PocoDataset, get_data_poco
from source.base.container import dict_np_to_torch
from source.base.proximity import query_kdtree


class PPSurfDataModule(PocoDataModule):

    def __init__(self, num_pts_local: int,
                 in_file, workers, use_ddp, padding_factor, seed, manifold_points,
                 patches_per_shape: typing.Optional[int], do_data_augmentation: bool, batch_size: int):
        super(PPSurfDataModule, self).__init__(
            use_ddp=use_ddp, workers=workers, in_file=in_file, patches_per_shape=patches_per_shape,
            do_data_augmentation=do_data_augmentation, batch_size=batch_size,
            padding_factor=padding_factor, seed=seed, manifold_points=manifold_points)
        self.num_pts_local = num_pts_local

    def make_dataset(
            self, in_file: typing.Union[str, list], reconstruction: bool, patches_per_shape: typing.Optional[int],
            do_data_augmentation: bool):

        if reconstruction:
            dataset = PPSurfReconstructionDataset(
                in_file=in_file,
                num_pts_local=self.num_pts_local,
                padding_factor=self.padding_factor,
                seed=self.seed,
                use_ddp=self.use_ddp,
            )
        else:
            dataset = PPSurfDataset(
                in_file=in_file,
                num_pts_local=self.num_pts_local,
                padding_factor=self.padding_factor,
                seed=self.seed,
                patches_per_shape=self.patches_per_shape,
                do_data_augmentation=do_data_augmentation,
                use_ddp=self.use_ddp,
                manifold_points=self.manifold_points,
            )
        return dataset


class PPSurfDataset(PocoDataset):

    def __init__(self, in_file, num_pts_local, padding_factor, seed, use_ddp,
                 manifold_points, patches_per_shape: typing.Optional[int], do_data_augmentation=True):
        super(PPSurfDataset, self).__init__(
            in_file=in_file, padding_factor=padding_factor, seed=seed,
            use_ddp=use_ddp, manifold_points=manifold_points,
            patches_per_shape=patches_per_shape, do_data_augmentation=do_data_augmentation)

        self.num_pts_local = num_pts_local

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the patch center
    def __getitem__(self, shape_id):
        shape_data, pts_ms_raw = self.load_shape_by_index(shape_id, return_kdtree=True)
        kdtree = shape_data.pop('kdtree')

        if self.do_data_augmentation:
            import trimesh
            # optionally always pick the same points for a given patch index (mainly for debugging)
            # self.rng.seed(42)
            rand_rot = trimesh.transformations.random_rotation_matrix(self.rng.rand(3))
            shape_data = self.augment_shape(shape_data, rand_rot)

        # must be after augmentation
        shape_data = PPSurfDataset.get_local_subsamples(shape_data, kdtree, pts_ms_raw, self.num_pts_local)

        pts_local_ps = self.normalize_patches(
            pts_local_ms=shape_data['pts_local_ms'], pts_query_ms=shape_data['pts_query_ms'])

        shape_data['pts_local_ps'] = pts_local_ps
        shape_data = dict_np_to_torch(shape_data)  # must be before poco part
        shape_data = get_data_poco(shape_data)
        return shape_data

    @staticmethod
    def get_local_subsamples(shape_data, kdtree, pts_raw_ms, num_pts_local):
        _, patch_pts_ids = query_kdtree(kdtree=kdtree, pts_query=shape_data['pts_query_ms'],
                                        k=num_pts_local, sqr_dists=True)
        patch_pts_ids = patch_pts_ids.astype(np.int64)
        shape_data['pts_local_ms'] = pts_raw_ms[patch_pts_ids]
        return shape_data
    
    @staticmethod
    def normalize_patches(pts_local_ms, pts_query_ms):
        patch_radius_ms = PPSurfDataset.get_patch_radii(pts_local_ms, pts_query_ms)
        pts_local_ps = PPSurfDataset.model_space_to_patch_space(
            pts_to_convert_ms=pts_local_ms, pts_patch_center_ms=pts_query_ms,
            patch_radius_ms=patch_radius_ms)
        return pts_local_ps

    @staticmethod
    def get_patch_radii(pts_patch: np.array, query_pts: np.array):
        if pts_patch.shape[1] == 0:
            patch_radius = 0.0
        elif pts_patch.shape == query_pts.shape:
            patch_radius = np.linalg.norm(pts_patch - query_pts, axis=0)
        else:
            from source.base.math import cartesian_dist
            dist = cartesian_dist(np.repeat(
                np.expand_dims(query_pts, axis=1), pts_patch.shape[1], axis=1), pts_patch, axis=2)
            patch_radius = np.max(dist, axis=-1)
        return patch_radius
    
    @staticmethod
    def model_space_to_patch_space(pts_to_convert_ms: np.array, pts_patch_center_ms: np.array,
                                   patch_radius_ms: typing.Union[float, np.ndarray]):

        pts_patch_center_ms_repeated = \
            np.repeat(np.expand_dims(pts_patch_center_ms, axis=1), pts_to_convert_ms.shape[-2], axis=-2)
        pts_patch_space = pts_to_convert_ms - pts_patch_center_ms_repeated
        patch_radius_ms_expanded = np.expand_dims(np.expand_dims(patch_radius_ms, axis=1), axis=2)
        patch_radius_ms_repeated = np.repeat(patch_radius_ms_expanded, pts_to_convert_ms.shape[-2], axis=-2)
        patch_radius_ms_repeated = np.repeat(patch_radius_ms_repeated, pts_to_convert_ms.shape[-1], axis=-1)
        pts_patch_space = pts_patch_space / patch_radius_ms_repeated
        return pts_patch_space
    

class PPSurfReconstructionDataset(PPSurfDataset):

    def __init__(self, in_file, num_pts_local, padding_factor, seed, use_ddp):

        super(PPSurfReconstructionDataset, self).__init__(
            in_file=in_file, num_pts_local=num_pts_local, padding_factor=padding_factor, seed=seed,
            use_ddp=use_ddp, manifold_points=None, patches_per_shape=None, do_data_augmentation=False)

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the patch center
    @overrides
    def __getitem__(self, shape_id):
        shape_data, pts_ms_raw = self.load_shape_by_index(shape_id, return_kdtree=False)
        shape_data['pts_raw_ms'] = pts_ms_raw  # collate issue for batch size > 1
        shape_data = dict_np_to_torch(shape_data)
        return shape_data
