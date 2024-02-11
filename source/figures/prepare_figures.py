import os
import shutil
import typing
import sys
sys.path.append(os.path.abspath('.'))

from source.base.mp import start_process_pool
from source.base.fs import call_necessary


def _copy_files(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(src):
        shutil.copy(src, dst)
    else:
        print('File not found: {}'.format(src))


def _get_vertex_distances(
        input_mesh_file: str, reference_mesh_file: str, output_mesh_file: str,
        min_vertex_count: typing.Union[int, None], distance_batch_size=10000):

    import numpy as np
    import trimesh
    from trimesh.base import Trimesh

    from source.base import proximity, fs

    in_mesh: Trimesh = trimesh.load(input_mesh_file)
    ref_mesh = trimesh.load(reference_mesh_file)

    if min_vertex_count is not None:
        while in_mesh.vertices.shape[0] < min_vertex_count:
            in_mesh = in_mesh.subdivide()

    closest_pts, dist_rec_verts_to_ref, tri_id = proximity.get_closest_point_on_mesh(
        mesh=ref_mesh, query_pts=in_mesh.vertices, batch_size=int(distance_batch_size))

    fs.make_dir_for_file(output_mesh_file)
    np.savez(file=output_mesh_file, vertices=in_mesh.vertices, faces=in_mesh.faces, distances=dist_rec_verts_to_ref)

    # debug output
    from trimesh.visual.color import VertexColor
    from source.base.visualization import distances_to_vertex_colors
    dist_cut_off = 0.1
    vertex_colors = distances_to_vertex_colors(dist_rec_verts_to_ref, dist_cut_off)
    in_mesh.visual = VertexColor(vertex_colors)
    in_mesh.export(output_mesh_file[:-4] + '_dist_col.ply')
    pass


def _assemble_figure_data(figure_path, objects, datasets_path, results_path, methods, workers=0):
    gt_in = [os.path.join(datasets_path, o[0], '03_meshes', o[1] + '.ply') for o in objects]
    gt_out = [os.path.join(figure_path, o[0], o[1], 'gt.ply') for o in objects]

    pc_in = [os.path.join(datasets_path, o[0], '04_pts_vis', o[1] + '.xyz.plys') for o in objects]
    pc_out = [os.path.join(figure_path, o[0], o[1], 'pc.ply') for o in objects]

    method_in = [[os.path.join(results_path, m, o[0], 'meshes', o[1] + '.ply') for m in methods] for o in objects]
    method_out = [[os.path.join(figure_path, o[0], o[1], m + '.ply') for m in methods] for o in objects]
    method_dist_out = [[os.path.join(figure_path, o[0], o[1], m + '_dist.npz') for m in methods] for o in objects]

    def _flatten(l):
        return [item for sublist in l for item in sublist]

    def _make_params(l1, l2):
        params = tuple(zip(l1, l2))
        params_necessary = [p for p in params if call_necessary(p[0], p[1], verbose=False)]
        return params_necessary

    start_process_pool(_copy_files, _make_params(gt_in, gt_out), num_processes=workers)
    start_process_pool(_copy_files, _make_params(_flatten(method_in), _flatten(method_out)), num_processes=workers)

    from source.base.point_cloud import numpy_to_ply
    start_process_pool(numpy_to_ply, _make_params(pc_in, pc_out), num_processes=workers)

    min_vertex_count = 10000
    distance_batch_size = 1000
    params = [tuple(zip(m, [gt_out[mi]] * len(m), method_dist_out[mi],
                        [min_vertex_count] * len(m), [distance_batch_size] * len(m)))
              for mi, m in enumerate(method_out)]
    params_flat = _flatten(params)
    params_flat_necessary = [p for p in params_flat if call_necessary((p[0], p[1]), p[2], verbose=False)]
    start_process_pool(_get_vertex_distances, params_flat_necessary, num_processes=workers)


if __name__ == '__main__':
    workers = 15  # for training PC
    # workers = 8  # for Windows missing fork
    # workers = 4  # for strange window size bug
    # workers = 0  # debug

    datasets_path = 'datasets'

    results_path = 'results'
    methods_comp = [
        'neural_imls',
        'pgr',
        'sap_optim',
        'sap',
        'p2s',
        'poco Pts_gen_sub3k_iter10',
        'ppsurf_merge_sum',
    ]

    figure_path_comp = 'results/figures/comp'
    objects_comp = [
        ('abc', '00010429_fc56088abf10474bba06f659_trimesh_004'),
        ('abc', '00011602_c087f04c99464bf7ab2380c4_trimesh_000'),
        ('abc', '00013052_9084b77631834dd584b2ac93_trimesh_033'),
        ('abc', '00014452_55263057b8f440a0bb50b260_trimesh_017'),
        ('abc', '00017014_fbef9df8f24940a0a2df6ccb_trimesh_001'),
        ('abc', '00990573_d1914c7f68f9a6b58bed9421_trimesh_000'),
        ('abc_noisefree', '00012754_b17656deace54b61b3130c7e_trimesh_019'),
        ('abc_noisefree', '00011696_1ca1ad2a09504ff1bf83cf74_trimesh_029'),
        ('abc_noisefree', '00016680_5a9a2a2a5eb64501863164e9_trimesh_000'),
        ('abc_noisefree', '00017682_f0ea0b827ae34675a4162390_trimesh_003'),
        ('abc_noisefree', '00019114_87f2e2e15b2746ffa4a2fd9a_trimesh_003'),
        ('abc_noisefree', '00011171_db6e2de6f4ae4ec493ebe2aa_trimesh_047'),
        ('abc_noisefree', '00011171_db6e2de6f4ae4ec493ebe2aa_trimesh_047'),
        ('abc_extra_noisy', '00013052_9084b77631834dd584b2ac93_trimesh_033'),
        ('abc_extra_noisy', '00014101_7b2cf2f0fd464e80a5062901_trimesh_000'),
        ('abc_extra_noisy', '00014155_a04f003ab9b74295bbed8248_trimesh_000'),
        ('abc_extra_noisy', '00016144_8dadc1c5885e427292f34e71_trimesh_026'),
        ('abc_extra_noisy', '00018947_b302da1a26764dd0afcd55ff_trimesh_075'),
        ('abc_extra_noisy', '00019203_1bcd132f82c84761b4e9851d_trimesh_001'),
        ('abc_extra_noisy', '00992690_ed0f9f06ad21b92e7ffab606_trimesh_002'),
        ('famous_dense', 'tortuga'),
        ('famous_dense', 'yoda'),
        ('famous_dense', 'armadillo'),
        ('famous_extra_noisy', 'Utah_teapot_(solid)'),
        ('famous_extra_noisy', 'happy'),
        ('famous_noisefree', 'galera'),
        ('famous_original', 'hand'),
        ('famous_original', 'horse'),
        ('famous_sparse', 'xyzrgb_statuette'),
        ('famous_sparse', 'dragon'),
        ('thingi10k_scans_dense', '58982'),
        ('thingi10k_scans_dense', '70558'),
        ('thingi10k_scans_dense', '77245'),
        ('thingi10k_scans_dense', '88053'),
        ('thingi10k_scans_extra_noisy', '86848'),
        ('thingi10k_scans_extra_noisy', '83022'),
        ('thingi10k_scans_noisefree', '103354'),
        ('thingi10k_scans_noisefree', '53159'),
        ('thingi10k_scans_noisefree', '54725'),
        ('thingi10k_scans_original', '53920'),
        ('thingi10k_scans_original', '64194'),
        ('thingi10k_scans_original', '73075'),
        ('thingi10k_scans_sparse', '80650'),
        ('thingi10k_scans_sparse', '81368'),
        ('thingi10k_scans_sparse', '81762'),
        ('real_world', 'madersperger_cropped'),
        ('real_world', 'statue_ps_outliers2_cropped'),
        # ('real_world', 'statue_ps_pointcleannet_cropped'),
        ('real_world', 'torch_ps_outliers2'),
    ]
    # for general comparison
    _assemble_figure_data(figure_path=figure_path_comp, methods=methods_comp, objects=objects_comp,
                          datasets_path=datasets_path, results_path=results_path, workers=workers)

    figure_path_ablation = 'results/figures/ablation'
    objects_ablation = [
        ('abc', '00012451_f54bcfcb352445bf90726b58_trimesh_001'),
        ('abc', '00014221_57e4213b31844b5b95cc62cd_trimesh_000'),
        ('abc', '00015159_57353d3381fb481182d9bdc6_trimesh_013'),
        ('abc', '00990546_db31ddca9d3585c330dcce3a_trimesh_000'),
        ('abc', '00993692_494894597fe7b39310a44a99_trimesh_000'),
    ]
    methods_ablation = [
        'ppsurf_vanilla_zeros_local',
        'ppsurf_vanilla_zeros_global',
        'ppsurf_vanilla_sym_max',
        'ppsurf_vanilla_qpoints',
        'ppsurf_vanilla',
        'ppsurf_merge_sum',
    ]
    # for ablation study
    _assemble_figure_data(figure_path=figure_path_ablation, methods=methods_ablation, objects=objects_ablation,
                          datasets_path=datasets_path, results_path=results_path, workers=workers)

    figure_path_real = 'results/figures/real_world'
    objects_real = [
        ('real_world', 'madersperger_cropped'),
        ('real_world', 'statue_ps_outliers2_cropped'),
        ('real_world', 'torch_ps_outliers2'),
    ]
    # for ablation study
    _assemble_figure_data(figure_path=figure_path_real, methods=methods_comp, objects=objects_real,
                          datasets_path=datasets_path, results_path=results_path, workers=workers)

    figure_path_dataset = 'results/figures/datasets'
    objects_dataset = [
        ('abc', '00013052_9084b77631834dd584b2ac93_trimesh_033'),
        ('abc_noisefree', '00013052_9084b77631834dd584b2ac93_trimesh_033'),
        ('abc_extra_noisy', '00013052_9084b77631834dd584b2ac93_trimesh_033'),
        ('famous_dense', 'hand'),
        ('famous_extra_noisy', 'hand'),
        ('famous_noisefree', 'hand'),
        ('famous_original', 'hand'),
        ('famous_sparse', 'hand'),
        ('thingi10k_scans_dense', '54725'),
        ('thingi10k_scans_extra_noisy', '54725'),
        ('thingi10k_scans_noisefree', '54725'),
        ('thingi10k_scans_original', '54725'),
        ('thingi10k_scans_sparse', '54725'),
    ]
    # for datasets figure
    _assemble_figure_data(figure_path=figure_path_dataset, methods=[], objects=objects_dataset,
                          datasets_path=datasets_path, results_path=results_path, workers=workers)

    figure_path_limitations = 'results/figures/limitations'
    objects_limitations = [
        ('thingi10k_scans_sparse', '274379'),
    ]
    # for limitations figure
    _assemble_figure_data(figure_path=figure_path_limitations, methods=['ppsurf_merge_sum'], objects=objects_limitations,
                          datasets_path=datasets_path, results_path=results_path, workers=workers)
