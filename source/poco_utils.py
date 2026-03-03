import typing
import os

import trimesh
import numpy as np
import torch
import torch.nn.functional as func
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar

import source.base.mesh

# Adapted from POCO: https://github.com/valeoai/POCO
# which is published under Apache 2.0: https://github.com/valeoai/POCO/blob/main/LICENSE


def profile_from_latent(func, data: dict):
    import time
    start = time.time()
    res: torch.Tensor = func(data)
    end = time.time()
    print('{} took: {}, shape: {}'.format('from_latent', end - start, res.shape))
    return res


def generate_latent_representation(batch, network, network_latent_size, gen_subsample_manifold_iter, gen_subsample_manifold, prog_bar: typing.Optional[TQDMProgressBar] = None):
    from source.poco_data_loader import get_data_poco
    
    if batch['pts_ms'].shape[0] > 1:
        raise NotImplementedError('batch size > 1 not supported')
    
    shape_data_poco = get_data_poco(batch_data=batch)

    pts = shape_data_poco['pts'][0].transpose(0, 1)

    # create the latent storage
    latent = torch.zeros((pts.shape[0], network_latent_size),
                            dtype=torch.float, device=pts.device)
    counts = torch.zeros((pts.shape[0],), dtype=torch.float, device=pts.device)

    iteration = 0
    for current_value in range(gen_subsample_manifold_iter):
        while counts.min() < current_value + 1:
            valid_ids = torch.argwhere(counts == current_value)[:, 0].clone().detach().long()

            if pts.shape[0] >= gen_subsample_manifold:
                ids = torch.randperm(valid_ids.shape[0])[:gen_subsample_manifold]
                ids = valid_ids[ids]

                if ids.shape[0] < gen_subsample_manifold:
                    ids = torch.cat(
                        [ids, torch.randperm(pts.shape[0], device=pts.device)[
                                :gen_subsample_manifold - ids.shape[0]]],
                        dim=0)
                assert (ids.shape[0] == gen_subsample_manifold)
            else:
                ids = torch.arange(pts.shape[0])

            data_partial = {'pts': shape_data_poco['pts'][0].transpose(1, 0)[ids].transpose(1, 0).unsqueeze(0)}
            partial_latent = network.get_latent(data_partial)['latents']
            latent[ids] += partial_latent[0].transpose(1, 0)
            counts[ids] += 1

            iteration += 1
            if prog_bar is not None:
                prog_bar.predict_progress_bar.set_postfix_str('get_latent iter: {}'.format(iteration), refresh=True)

    latent = latent / counts.unsqueeze(1)
    latent = latent.transpose(1, 0).unsqueeze(0)
    shape_data_poco['latents'] = latent
    latent = shape_data_poco
    return shape_data_poco, latent 

def get_pts_local_ps(latent: dict, pts_query: np.ndarray, kdtree, pts_raw_ms, num_pts_local):
    from source.base.proximity import query_kdtree
    from source.ppsurf_data_loader import PPSurfDataset
    
    _, patch_pts_ids = query_kdtree(kdtree=kdtree, pts_query=pts_query, k=num_pts_local, sqr_dists=True)
    pts_local_ms = pts_raw_ms[patch_pts_ids.astype(np.int64)]
    pts_local_ps_np = PPSurfDataset.normalize_patches(pts_local_ms=pts_local_ms, pts_query_ms=pts_query)
    pts_local_ps = torch.from_numpy(pts_local_ps_np).to(latent['pts_ms'].device).unsqueeze(0)
    return pts_local_ps

def predict_from_latent(latent: dict, network, pts_query, num_pts_local, kdtree=None):
    
    latent['pts_query'] = pts_query.unsqueeze(0)
    if num_pts_local is not None and num_pts_local > 0:
        latent['pts_local_ps'] = get_pts_local_ps(latent=latent, pts_query=pts_query.detach().cpu().numpy(), kdtree=kdtree, pts_raw_ms=latent.get('pts_raw_ms'), num_pts_local=num_pts_local)
    
    occ_hat = network.from_latent(latent)
    # occ_hat = profile_from_latent(network.from_latent, _latent)

    # get class and non-class
    occ_hat = func.softmax(occ_hat, dim=1)
    occ_hat = occ_hat[:, 0] - occ_hat[:, 1]
    occ_hat = occ_hat.squeeze(0).detach().cpu().numpy()
    return occ_hat
    
def export_mesh_and_refine_vertices_region_growing_v3(
        network: pl.LightningModule,
        latent: dict,
        pts_raw_ms: torch.Tensor,
        resolution: int,
        input_points,
        padding=0,
        mc_value=0,
        num_pts=50000,
        num_pts_local=None,
        refine_iter=10,
        out_value=np.nan,
        dilation_size=2,
        prog_bar: typing.Optional[TQDMProgressBar] = None,
        pc_file_in: str = 'unknown',
) -> typing.Optional[trimesh.Trimesh]:
    from skimage import measure
    from source.base.proximity import make_kdtree

    if latent['pts_ms'].shape[0] != 1:
        raise ValueError('Reconstruction must be done with batch size = 0!')

    bmin = input_points.min()
    bmax = input_points.max()

    step = (bmax - bmin) / (resolution - 1)

    bmin_pad = bmin - padding * step
    bmax_pad = bmax + padding * step

    pts_ids = (input_points - bmin) / step + padding
    pts_ids = pts_ids.astype(np.int32)

    if num_pts_local is not None:
        pts_raw_ms = pts_raw_ms[0].detach().cpu().numpy()  # expect batch size = 1
        kdtree = make_kdtree(pts=pts_raw_ms)
    else:
        kdtree = None

    volume = _create_volume(network, dilation_size, bmin_pad, latent, num_pts,
                            num_pts_local, out_value, padding, pc_file_in, prog_bar, pts_ids, resolution, step, kdtree=kdtree)

    # volume[np.isnan(volume)] = out_value
    maxi = volume[~np.isnan(volume)].max()
    mini = volume[~np.isnan(volume)].min()

    # occ doesn't cross zero-level set
    if not (maxi > mc_value > mini):
        return None

    # compute the marching cubes
    verts, faces, _, _ = measure.marching_cubes(volume=volume.copy(), level=mc_value)

    # remove the nan values in the vertices
    # values = verts.sum(axis=1)
    # invalid_vertices_mask = np.isnan(values)
    # verts = np.asarray(verts[invalid_vertices_mask])
    # faces = np.asarray(faces)

    # clean mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    source.base.mesh.clean_simple_inplace(mesh=mesh)
    mesh = source.base.mesh.remove_small_connected_components(mesh=mesh, num_faces=6)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if refine_iter > 0:
        verts = refine_vertices(network, latent, num_pts, num_pts_local, refine_iter, prog_bar, pc_file_in, step, bmin_pad, get_pts_local_ps, volume, verts)
    else:
        verts = verts * step + bmin_pad

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    source.base.mesh.clean_simple_inplace(mesh=mesh)
    mesh = source.base.mesh.remove_small_connected_components(mesh=mesh, num_faces=6)
    return mesh

def refine_vertices(network, latent, num_pts, num_pts_local, refine_iter, prog_bar, pc_file_in, step, bmin_pad, _get_pts_local_ps, volume, verts, kdtree=None):
    from tqdm import tqdm
    
    dirs = verts - np.floor(verts)
    dirs = (dirs > 0).astype(dirs.dtype)

    mask = np.logical_and(dirs.sum(axis=1) > 0, dirs.sum(axis=1) < 2)
    v = verts[mask]
    dirs = dirs[mask]

        # initialize the two values (the two vertices for mc grid)
    v1 = np.floor(v)
    v2 = v1 + dirs

        # get the predicted values for both set of points
    v1 = v1.astype(int)
    v2 = v2.astype(int)
    preds1 = volume[v1[:, 0], v1[:, 1], v1[:, 2]]
    preds2 = volume[v2[:, 0], v2[:, 1], v2[:, 2]]

        # get the coordinates in the real coordinate system
    v1 = v1.astype(np.float32) * step + bmin_pad
    v2 = v2.astype(np.float32) * step + bmin_pad

        # tmp mask
    mask_tmp = np.logical_and(np.logical_not(np.isnan(preds1)), np.logical_not(np.isnan(preds2)))
    v = v[mask_tmp]
        # dirs = dirs[mask_tmp]
    v1 = v1[mask_tmp]
    v2 = v2[mask_tmp]
    mask[mask] = mask_tmp

        # initialize the vertices
    verts = verts * step + bmin_pad
    v = v * step + bmin_pad

        # iterate for the refinement step
    for iter_id in range(refine_iter):
        preds = []
        pnts_all = torch.tensor(v, dtype=torch.float)
        for pts_query in tqdm(torch.split(pnts_all, num_pts, dim=0), ncols=100, disable=True):
            latent['pts_query'] = pts_query.unsqueeze(0)
            if num_pts_local is not None:
                latent['pts_local_ps'] = _get_pts_local_ps(latent=latent, pts_query=pts_query.detach().cpu().numpy(), kdtree=kdtree, pts_raw_ms=latent.get('pts_raw_ms'), num_pts_local=num_pts_local)
            preds.append(predict_from_latent(latent, network, pts_query, num_pts_local, kdtree))
        preds = np.concatenate(preds, axis=0)

        mask1 = (preds * preds1) > 0
        v1[mask1] = v[mask1]
        preds1[mask1] = preds[mask1]

        mask2 = (preds * preds2) > 0
        v2[mask2] = v[mask2]
        preds2[mask2] = preds[mask2]

        v = (v2 + v1) / 2
        verts[mask] = v

        prog_bar.predict_progress_bar.set_postfix_str('{}, refine iter {}'.format(
                os.path.basename(pc_file_in)[:16], iter_id), refresh=True)
            
    return verts


def _create_volume(network, dilation_size, bmin_pad, latent, num_pts, num_pts_local,
                   out_value, padding, pc_file_in, prog_bar, pts_ids, resolution, step, kdtree=None):

    def _dilate_binary(arr: np.ndarray, pts_int: np.ndarray):
        # old POCO version actually dilates with a 4^3 kernel, 2 to lower, 1 to upper
        # -> no out-of upper bounds with 2 dilation_size by default
        # we make it symmetric (+1 to max)
        pts_min = np.maximum(0, pts_int - dilation_size)
        pts_max = np.minimum(arr.shape[0], pts_int + dilation_size + 1)

        def _dilate_point(pt_min, pt_max):
            arr[pt_min[0]:pt_max[0],
            pt_min[1]:pt_max[1],
            pt_min[2]:pt_max[2]] = True

        # vectorizing slices is not possible? so we iterate over the points
        # skimage.morphology and scipy.ndimage take longer, probably because of overhead
        _ = [_dilate_point(pt_min=pts_min[i], pt_max=pts_max[i]) for i in range(pts_int.shape[0])]
        return arr

    res_x = resolution
    res_y = resolution
    res_z = resolution

    volume_shape = (res_x + 2 * padding, res_y + 2 * padding, res_z + 2 * padding)
    volume = np.full(volume_shape, np.nan, dtype=np.float64)
    mask_to_see = np.full(volume_shape, True, dtype=bool)
    while pts_ids.shape[0] > 0:
        # create the mask
        mask = np.full(volume_shape, False, dtype=bool)
        mask[pts_ids[:, 0], pts_ids[:, 1], pts_ids[:, 2]] = True
        mask = _dilate_binary(arr=mask, pts_int=pts_ids)

        # get the valid points
        valid_points_coord = np.argwhere(mask).astype(np.float32)
        valid_points = valid_points_coord * step + bmin_pad

        # get the prediction for each valid points
        z = []
        near_surface_samples_torch = torch.tensor(valid_points, dtype=torch.float)
        for pts_query in torch.split(near_surface_samples_torch, num_pts, dim=0):

            latent['pts_query'] = pts_query.unsqueeze(0)
            if num_pts_local is not None:
                latent['pts_local_ps'] = get_pts_local_ps(latent=latent, pts_query=pts_query.detach().cpu().numpy(), kdtree=kdtree, pts_raw_ms=latent.get('pts_raw_ms'), num_pts_local=num_pts_local)
            z.append(predict_from_latent(latent, network, pts_query, num_pts_local, kdtree))

            prog_bar.predict_progress_bar.set_postfix_str(
                '{}, occ_batch iter {}'.format(os.path.basename(pc_file_in), len(z)), refresh=True)

        z = np.concatenate(z, axis=0)
        z = z.astype(np.float64)

        # update the volume
        volume[mask] = z

        # create the masks
        mask_pos = np.full(volume_shape, False, dtype=bool)
        mask_neg = np.full(volume_shape, False, dtype=bool)
        mask_to_see[pts_ids[:, 0], pts_ids[:, 1], pts_ids[:, 2]] = False

        # dilate
        pts_ids_pos = pts_ids[volume[pts_ids[:, 0], pts_ids[:, 1], pts_ids[:, 2]] <= 0]
        pts_ids_neg = pts_ids[volume[pts_ids[:, 0], pts_ids[:, 1], pts_ids[:, 2]] >= 0]
        mask_neg = _dilate_binary(arr=mask_neg, pts_int=pts_ids_pos)
        mask_pos = _dilate_binary(arr=mask_pos, pts_int=pts_ids_neg)

        # get the new points
        new_mask = (mask_neg & (volume >= 0) & mask_to_see) | (mask_pos & (volume <= 0) & mask_to_see)
        pts_ids = np.argwhere(new_mask).astype(np.int64)
    volume[0:padding, :, :] = out_value
    volume[-padding:, :, :] = out_value
    volume[:, 0:padding, :] = out_value
    volume[:, -padding:, :] = out_value
    volume[:, :, 0:padding] = out_value
    volume[:, :, -padding:] = out_value
    return volume


@torch.jit.ignore
def knn(points: torch.Tensor, support_points: torch.Tensor, k: int, workers: int = 1) -> torch.Tensor:
    if k > points.shape[2]:
        k = points.shape[2]
    pts = points.cpu().detach().transpose(1, 2).numpy().copy()
    s_pts = support_points.cpu().detach().transpose(1, 2).numpy().copy()

    from source.base.proximity import kdtree_query_oneshot

    indices: list = []
    for i in range(pts.shape[0]):
        _, ids = kdtree_query_oneshot(pts=pts[i], pts_query=s_pts[i], k=k, workers=workers)
        indices.append(torch.from_numpy(ids.astype(np.int64)))
    indices: torch.Tensor = torch.stack(indices, dim=0)
    if k == 1:
        indices = indices.unsqueeze(2)
    return indices.to(points.device)
