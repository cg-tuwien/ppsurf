import typing

import numpy as np
import trimesh


def clean_simple_inplace(mesh: trimesh.Trimesh):
    # extra function because this overlaps only partially with mesh.process(validate=True)
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.merge_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()


def remove_small_connected_components(mesh: trimesh.Trimesh, num_faces: typing.Optional[int] = 100):
    from trimesh import graph

    # https://github.com/Wuziyi616/IF-Defense/blob/main/ONet/data_proc/make_watertight.py

    # if num_faces not given, take 1 % of faces
    total_num_faces = len(mesh.faces)
    if num_faces is None:
        num_faces = total_num_faces // 100

    cc = graph.connected_components(mesh.face_adjacency, min_len=3)
    mask = np.zeros(total_num_faces, dtype=bool)
    cc_large_enough = [c for c in cc if len(c) > num_faces]
    if len(cc_large_enough) == 0:
        cc_large_enough = np.empty()
    cc = np.concatenate(cc_large_enough, axis=0)
    mask[cc] = True
    mesh.update_faces(mask)

    # clean to keep only used data
    clean_simple_inplace(mesh=mesh)

    return mesh
