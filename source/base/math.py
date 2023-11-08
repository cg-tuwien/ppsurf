import typing

import numpy as np
import trimesh


def cartesian_dist_1_n(vec_x: np.array, vec_y: np.array, axis=1) -> np.ndarray:
    """
    L2 distance
    :param vec_x: array[d]
    :param vec_y: array[n, d]
    :param axis: int
    :return: array[n]
    """
    vec_x_bc = np.tile(np.expand_dims(vec_x, 0), (vec_y.shape[0], 1))

    dist = np.linalg.norm(vec_x_bc - vec_y, axis=axis)
    return dist


def cartesian_dist(vec_x: np.array, vec_y: np.array, axis=1) -> np.ndarray:
    """
    L2 distance
    :param vec_x: array[n, d]
    :param vec_y: array[n, d]
    :param axis: int
    :return: array[n]
    """
    dist = np.linalg.norm(vec_x - vec_y, axis=axis)
    return dist


def vector_length(vecs: np.array, axis=1) -> np.ndarray:
    dist = np.linalg.norm(vecs, axis=axis)
    return dist


def normalize_vectors(vecs: np.array):
    """
    :param vecs: array[n, dims]
    :return:
    """
    n_dims = vecs.shape[1]
    vecs_normalized = vecs / np.repeat(vector_length(vecs)[:, np.newaxis], repeats=n_dims, axis=1)
    return vecs_normalized


def get_patch_radii(pts_patch: np.array, query_pts: np.array):
    if pts_patch.shape[0] == 0:
        patch_radius = 0.0
    elif pts_patch.shape == query_pts.shape:
        patch_radius = np.linalg.norm(pts_patch - query_pts, axis=0)
    else:
        dist = cartesian_dist(np.repeat(np.expand_dims(query_pts, axis=0), pts_patch.shape[0], axis=0),
                              pts_patch, axis=1)
        patch_radius = np.max(dist, axis=-1)
    return patch_radius


def model_space_to_patch_space_single_point(
        pts_to_convert_ms: np.array, pts_patch_center_ms: np.array, patch_radius_ms: typing.Union[float, np.ndarray]):

    pts_patch_space = pts_to_convert_ms - pts_patch_center_ms
    pts_patch_space = pts_patch_space / patch_radius_ms
    return pts_patch_space


def model_space_to_patch_space(
        pts_to_convert_ms: np.array, pts_patch_center_ms: np.array, patch_radius_ms: typing.Union[float, np.ndarray]):

    pts_patch_center_ms_repeated = \
        np.repeat(np.expand_dims(pts_patch_center_ms, axis=0), pts_to_convert_ms.shape[-2], axis=-2)
    pts_patch_space = pts_to_convert_ms - pts_patch_center_ms_repeated
    pts_patch_space = pts_patch_space / patch_radius_ms

    return pts_patch_space


def lerp(
        a: np.ndarray,
        b: np.ndarray,
        factor: typing.Union[np.ndarray, float]):
    interpolated = a + factor * (b - a)
    return interpolated


def normalize_data(arr: np.ndarray, in_max: float, in_min: float, out_max=1.0, out_min=-1.0, clip=False):

    arr = arr.copy()
    in_range = in_max - in_min
    out_range = out_max - out_min

    if in_range == 0.0 or out_range == 0.0:
        print('Warning: normalization would result in NaN, kept raw values')
        return arr - in_max

    # scale so that in_max=1.0 and in_min=0.0
    arr -= in_min
    arr /= in_range

    # scale to out_max..out_min
    arr *= out_range
    arr += out_min

    if clip:
        arr = np.clip(arr, out_min, out_max)

    return arr


def get_points_normalization_info(pts: np.ndarray, padding_factor: float = 0.05):
    pts_bb_min = np.min(pts, axis=0)
    pts_bb_max = np.max(pts, axis=0)

    bb_center = (pts_bb_min + pts_bb_max) * 0.5
    scale = np.max(pts_bb_max - pts_bb_min) * (1.0 + padding_factor)
    return bb_center, scale


def normalize_points_with_info(pts: np.ndarray, bb_center: np.ndarray, scale: float):
    pts_new = pts - np.tile(bb_center, reps=(pts.shape[0], 1))
    pts_new /= scale

    # pts_new = pts / scale
    # pts_new -= np.tile(-bb_center, reps=(pts.shape[0], 1))
    return pts_new


def denormalize_points_with_info(pts: np.ndarray, bb_center: np.ndarray, scale: float):
    pts_new = pts * scale
    pts_new += np.tile(bb_center, reps=(pts.shape[0], 1))
    return pts_new


def rotate_points_around_pivot(pts: np.ndarray, rotation_mat: np.ndarray, pivot: np.ndarray):
    """
    rotate_points_around_pivot
    :param pts: np.ndarray[n, dims=3]
    :param rotation_mat: np.ndarray[4, 4]
    :param pivot: np.ndarray[dims=3]
    :return:
    """
    pivot_bc = np.broadcast_to(pivot[np.newaxis, :], pts.shape)

    pts -= pivot_bc
    pts = trimesh.transformations.transform_points(pts, rotation_mat)
    pts += pivot_bc

    return pts


def _test_normalize():
    ms = 0.75
    vs = 1.0 / 32
    # padding_factor = 0.0
    padding_factor = 0.05
    pts_ms = np.array([[-ms, -ms], [-ms, +ms], [+ms, -ms], [+ms, +ms], [0.0, 0.0],
                       [vs*0.3, -vs*0.3], [vs*0.5, -vs*0.5], [vs*0.6, -vs*0.6]])
    pts_ms *= 76.0
    pts_ms += 123.0

    # vertices = np.random.random(size=(25, 2)) * 2.0 - 1.0
    vertices = pts_ms

    bb_center, scale = get_points_normalization_info(pts=pts_ms, padding_factor=padding_factor)
    vertices_norm = normalize_points_with_info(pts=vertices, bb_center=bb_center, scale=scale)
    vertices_denorm = denormalize_points_with_info(pts=vertices_norm, bb_center=bb_center, scale=scale)

    if not np.allclose(vertices_denorm, vertices):
        raise ValueError()

    if vertices_norm.max() > 0.5 or vertices_norm.min() < -0.5:
        raise ValueError()

    return 0


if __name__ == '__main__':
    _test_normalize()
