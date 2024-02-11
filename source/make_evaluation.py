import os
import argparse
import sys

sys.path.append(os.path.abspath('.'))


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='ppsurf', help='name')

    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loading workers - 0 means same thread as main execution')

    parser.add_argument('--results_dir', type=str, default='results',
                        help='output folder (reconstructions)')
    parser.add_argument('--data_dir', type=str, default='datasets/abc_minimal/03_meshes',
                        help='input folder (meshes)')
    parser.add_argument('--testset', type=str, default='datasets/abc_minimal/testset.txt',
                        help='test set file name')

    parser.add_argument('--num_samples', type=int, default=10000,
                        help='number of samples for metrics')

    return parser.parse_args(args=args)


def make_evaluation(args):
    from source.base import evaluation
    from source.occupancy_data_module import read_shape_list

    model_results_rec_dir = os.path.join(args.results_dir, args.name, os.path.basename(args.data_dir))
    shape_names = read_shape_list(os.path.join(args.data_dir, args.testset))
    gt_meshes_dir = os.path.join(args.data_dir, '03_meshes')
    if not os.path.exists(gt_meshes_dir):
        print('Warning: {} not found. Skipping evaluation.'.format(gt_meshes_dir))
    else:
        gt_meshes = [os.path.join(gt_meshes_dir, '{}.ply'.format(vs)) for vs in shape_names]
        os.makedirs(model_results_rec_dir, exist_ok=True)
        result_headers = [args.name]
        result_file_templates = [os.path.join(model_results_rec_dir, 'meshes/{}.xyz.ply')]
        _ = evaluation.make_quantitative_comparison(
            shape_names=shape_names, gt_mesh_files=gt_meshes,
            result_headers=result_headers, result_file_templates=result_file_templates,
            comp_output_dir=model_results_rec_dir, num_processes=args.workers, num_samples=args.num_samples)


def main(argv=None):
    args = parse_arguments(argv)
    make_evaluation(args=args)


if __name__ == '__main__':
    # main()

    # test
    model_names = [
        'pgr',
        'neural_imls',
        'sap_optim',
        'sap',
        'p2s',
        'poco Pts_gen_sub3k_iter10',
        'ppsurf_qpoints',
        'ppsurf_merge_sum',
        'ppsurf_vanilla_zeros_local',
        'ppsurf_vanilla_zeros_global',
        'ppsurf_10nn',
        'ppsurf_25nn',
        'ppsurf_50nn',
        'ppsurf_100nn',
        'ppsurf_200nn',
    ]
    dataset_names = [
        'abc',
        'abc_extra_noisy',
        'abc_noisefree',
        # 'real_world',
        'famous_original',
        'famous_noisefree',
        'famous_sparse',
        'famous_dense',
        'famous_extra_noisy',
        'thingi10k_scans_original',
        'thingi10k_scans_noisefree',
        'thingi10k_scans_sparse',
        'thingi10k_scans_dense',
        'thingi10k_scans_extra_noisy'
    ]
    for dataset_name in dataset_names:
        for model_name in model_names:
            print('Evaluating {} on {}'.format(model_name, dataset_name))
            params = [
                '--name', model_name,
                '--workers', '15',
                # '--workers', '0',
                '--results_dir', 'results',
                '--data_dir', 'datasets/{}'.format(dataset_name),
                '--testset', 'testset.txt',
                '--num_samples', '100000',
            ]
            main(argv=params)
