import typing

import numpy as np
import trimesh
from torch.jit import ignore
from pysdf import SDF
from scipy.spatial import KDTree as ScipyKDTree
from pykdtree.kdtree import KDTree as PyKDTree


def get_signed_distance_pysdf_inaccurate(in_mesh: trimesh.Trimesh, query_pts_ms: np.ndarray):
    # pysdf is inaccurate up to +-0.1 of BB
    # this is not good enough for P2S at resolution > 64
    # but the sign is correct
    sdf = SDF(in_mesh.vertices, in_mesh.faces)
    dists_ms = sdf(query_pts_ms)
    return dists_ms


def get_closest_point_on_mesh(mesh: trimesh.Trimesh, query_pts, batch_size=1000):

    import trimesh.proximity as prox

    # process batches because trimesh's closest_point very inefficient on memory, similar to signed_distance()
    closest_pts = np.zeros((query_pts.shape[0], 3), dtype=np.float32)
    dists = np.zeros(query_pts.shape[0], dtype=np.float32)
    tri_ids = np.zeros(query_pts.shape[0], dtype=np.int32)
    pts_ids = np.arange(query_pts.shape[0])
    pts_ids_split = np.array_split(pts_ids, max(1, int(query_pts.shape[0] / batch_size)))
    for pts_ids_batch in pts_ids_split:
        query_pts_batch = query_pts[pts_ids_batch]
        closest_pts_batch, dist_batch, tri_id_batch = prox.closest_point(mesh=mesh, points=query_pts_batch)
        closest_pts[pts_ids_batch] = closest_pts_batch
        dists[pts_ids_batch] = dist_batch
        tri_ids[pts_ids_batch] = tri_id_batch

    return closest_pts, dists, tri_ids


def make_kdtree(pts: np.ndarray):

    # old reliable
    def _make_kdtree_scipy(pts_np: np.ndarray):
        # otherwise KDTree construction may run out of recursions
        import sys
        leaf_size = 1000
        sys.setrecursionlimit(int(max(1000, round(pts_np.shape[0] / leaf_size))))
        _kdtree = ScipyKDTree(pts_np, leaf_size)
        return _kdtree

    # a lot slower than scipy
    # def _make_kdtree_sklearn(pts: np.ndarray):
    #     from sklearn.neighbors import NearestNeighbors
    #     nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=workers).fit(pts)
    #     # indices_batch: np.ndarray = nbrs.kneighbors(pts_query_np, return_distance=False)

    # fastest even without multiprocessing
    def _make_kdtree_pykdtree(pts_np: np.ndarray):
        _kdtree = PyKDTree(pts_np, leafsize=10)
        return _kdtree

    # kdtree = _make_kdtree_scipy(pts)
    kdtree = _make_kdtree_pykdtree(pts)
    return kdtree


def query_kdtree(kdtree: typing.Union[ScipyKDTree, PyKDTree],
                 pts_query: np.ndarray, k: int, sqr_dists=False, **kwargs):
    # sqr_dists: some speed-up if True but distorted distances

    if isinstance(kdtree, ScipyKDTree):
        kdtree = typing.cast(ScipyKDTree, kdtree)
        nn_dists, nn_ids = kdtree.query(x=pts_query, k=k, workers=kwargs.get('workers', -1))
        if not sqr_dists:
            nn_dists = nn_dists ** 2
    elif isinstance(kdtree, PyKDTree):
        kdtree = typing.cast(PyKDTree, kdtree)
        nn_dists, nn_ids = kdtree.query(pts_query, k=k, sqr_dists=sqr_dists)
    else:
        raise NotImplementedError('Unknown kdtree type: {}'.format(type(kdtree)))
    return nn_dists, nn_ids


@ignore  # can't compile kdtree
def kdtree_query_oneshot(pts: np.ndarray, pts_query: np.ndarray, k: int, sqr_dists=False, **kwargs):
    # sqr_dists: some speed-up if True but distorted distances
    kdtree = make_kdtree(pts)
    nn_dists, nn_ids = query_kdtree(kdtree=kdtree, pts_query=pts_query, k=k, sqr_dists=sqr_dists, **kwargs)
    return nn_dists, nn_ids
