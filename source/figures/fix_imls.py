import os
import sys
sys.path.append(os.path.abspath('.'))

from source.base.fs import call_necessary, make_dir_for_file
from source.base.mp import start_process_pool


def _revert_normalization(src, gt, dst):
    import trimesh

    if not os.path.isfile(src):
        print('File not found: {}'.format(src))
        return

    mesh_gt = trimesh.load(gt)
    bounds = mesh_gt.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh_gt.bounds[0] + mesh_gt.bounds[1]) * 0.5
    translation_inv = trimesh.transformations.translation_matrix(direction=translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo_inv = trimesh.transformations.scale_matrix(factor=1.0 / scale)

    mesh_rec = trimesh.load(src)

    mesh_rec.apply_transform(scale_trafo_inv)
    mesh_rec.apply_transform(translation_inv)

    make_dir_for_file(dst)
    mesh_rec.export(dst)


if __name__ == '__main__':
    workers = 15  # for training PC
    # workers = 8  # for Windows missing fork
    # workers = 4  # for strange window size bug
    # workers = 0  # debug

    datasets_path = 'datasets'
    # extra-noisy is not provided
    datasets = [
        'abc',
        # 'abc_extra_noisy',
        'abc_noisefree',
        'famous_noisefree',
        'famous_original',
        # 'famous_extra_noisy',
        'famous_sparse',
        'famous_dense',
        'thingi10k_scans_original',
        'thingi10k_scans_dense',
        'thingi10k_scans_sparse',
        # 'thingi10k_scans_extra_noisy',
        'thingi10k_scans_noisefree',
    ]
    results_path = 'results'

    for d in datasets:
        test_set = os.path.join(datasets_path, d, 'testset.txt')
        test_shapes = [l.strip() for l in open(test_set, 'r').readlines()]
        test_files = [os.path.join(datasets_path, d, '03_meshes', s + '.ply') for s in test_shapes]

        rec_meshes_in = [os.path.join(results_path, 'neural_imls misaligned', d, 'meshes', s + '.ply') for s in test_shapes]
        rec_meshes_out = [os.path.join(results_path, 'neural_imls', d, 'meshes', s + '.ply') for s in test_shapes]

        def _make_params(l1, l2, l3):
            params = tuple(zip(l1, l2, l3))
            params_necessary = [p for p in params if call_necessary((p[0], p[1]), p[2], verbose=False)]
            return params_necessary

        start_process_pool(_revert_normalization, _make_params(rec_meshes_in, test_files, rec_meshes_out),
                           num_processes=workers)
