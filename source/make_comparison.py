import os
import argparse
import sys

import source.base.visualization
import source.occupancy_data_module

sys.path.append(os.path.abspath('.'))

debug = False


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--comp_name', type=str, default='abc_minimal', help='comp name')

    parser.add_argument('--comp_dir', type=str, default='results/comp', help='folder for comparisons')
    parser.add_argument('--data_dir', type=str, default='datasets/abc_minimal/03_meshes',
                        help='input folder (meshes)')
    parser.add_argument('--testset', type=str, default='datasets/abc_minimal/testset.txt',
                        help='test set file name')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='output folder (reconstructions)')
    parser.add_argument('--result_headers', type=str, nargs='+', default=[],
                        help='list of strings for comparison (human readable table headers)')
    parser.add_argument('--result_paths', type=str, nargs='+', default=[],
                        help='list of strings for comparison (result path templates)')
    parser.add_argument('--comp_mean_name', type=str, default='comp_mean',
                        help='file name for dataset means')
    parser.add_argument('--html_name', type=str, default='comp_html',
                        help='file name for dataset means')

    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loading workers - 0 means same thread as main execution')

    parser.add_argument('--dist_cut_off', type=float, default=0.05,
                        help='cutoff for color-coded distance visualization')

    return parser.parse_args(args=args)


def comparison_rec_mesh_template(args):
    from source.base import evaluation
    from itertools import chain
    from source.occupancy_data_module import read_shape_list

    comp_dir = os.path.join(args.comp_dir, args.comp_name)
    os.makedirs(comp_dir, exist_ok=True)

    shape_names = read_shape_list(os.path.join(args.data_dir, args.testset))
    gt_meshes = [os.path.join(args.data_dir, '03_meshes', '{}.ply'.format(vs)) for vs in shape_names]

    # quantitative comparison
    report_path_templates = [os.path.join(r, '{}.xlsx') for r in args.result_paths]
    results_per_shape_dict = evaluation.assemble_quantitative_comparison(
        comp_output_dir=comp_dir, report_path_templates=report_path_templates)
    cd_results = results_per_shape_dict['chamfer_distance'].transpose().tolist()
    iou_results = results_per_shape_dict['iou'].transpose().tolist()
    nc_results = results_per_shape_dict['normal_error'].transpose().tolist()

    # assemble dataset means
    # def _get_all_reports(results_dir, results_report_template):
    #     from pathlib import Path
    #     report_files = list(Path(results_dir).rglob(results_report_template))
    #     return report_files
    test_report_comp = os.path.join(comp_dir, '{}.xlsx'.format(args.comp_mean_name))
    cd_report_path = [os.path.join(r, 'chamfer_distance.xlsx') for r in args.result_paths]
    f1_report_path = [os.path.join(r, 'f1.xlsx') for r in args.result_paths]
    iou_report_path = [os.path.join(r, 'iou.xlsx') for r in args.result_paths]
    nc_report_path = [os.path.join(r, 'normal_error.xlsx') for r in args.result_paths]
    report_path_templates = [(cd_report_path[i], iou_report_path[i], f1_report_path[i], nc_report_path[i])
                             for i in range(len(args.result_paths))]
    evaluation.make_dataset_comparison(results_reports=report_path_templates, output_file=test_report_comp)

    # visualize chamfer distance as vertex colors
    gt_meshes_bc = [gt_meshes] * len(args.result_paths)
    gt_meshes_bc_flat = list(chain.from_iterable(gt_meshes_bc))
    cd_meshes_out = [[os.path.join(comp_dir, res, 'mesh_cd_vis', '{}.ply'.format(s))
                      for s in shape_names] for res in args.result_headers]
    cd_meshes_out_flat = list(chain.from_iterable(cd_meshes_out))
    rec_paths = [os.path.join(res, 'meshes/{}.xyz.ply') for res in args.result_paths]
    rec_meshes = [[res.format(s) for s in shape_names] for res in rec_paths]
    rec_meshes = [[s if os.path.isfile(s) else s[:-4] + '.obj' for s in res] for res in rec_meshes]  # if no ply, try obj
    rec_meshes_flat = list(chain.from_iterable(rec_meshes))
    source.base.visualization.visualize_chamfer_distance_pool(
        rec_meshes=rec_meshes_flat, gt_meshes=gt_meshes_bc_flat, output_mesh_files=cd_meshes_out_flat,
        min_vertex_count=10000, dist_cut_off=args.dist_cut_off, distance_batch_size=1000, num_processes=args.workers)

    # render meshes
    gt_renders_out = [os.path.join(comp_dir, 'mesh_gt_rend', '{}.png'.format(vs)) for vs in shape_names]
    rec_renders_out = [[os.path.join(comp_dir, res, 'mesh_rend', '{}.png'.format(s))
                        for s in shape_names] for res in args.result_headers]
    cd_vis_renders_out = [[os.path.join(comp_dir, res, 'cd_vis_rend', '{}.png'.format(s))
                           for s in shape_names] for res in args.result_headers]
    cd_vis_renders_out_flat = list(chain.from_iterable(cd_vis_renders_out))
    rec_renders_flat = list(chain.from_iterable(rec_renders_out))
    pc = [os.path.join(args.data_dir, '04_pts_vis', '{}.xyz.ply'.format(vs)) for vs in shape_names]
    pc_renders_out = [os.path.join(comp_dir, 'pc_rend', '{}.png'.format(vs)) for vs in shape_names]
    all_meshes_in = rec_meshes_flat + gt_meshes + cd_meshes_out_flat + pc
    all_renders_out = rec_renders_flat + gt_renders_out + cd_vis_renders_out_flat + pc_renders_out
    source.base.visualization.render_meshes(all_meshes_in, all_renders_out, workers=args.workers)

    # qualitative comparison as a HTML table
    report_file_out = os.path.join(comp_dir, args.html_name + '.html')
    evaluation.make_html_report(report_file_out=report_file_out, comp_name=args.comp_name,
                                pc_renders=pc_renders_out, gt_renders=gt_renders_out,
                                cd_vis_renders=cd_vis_renders_out, dist_cut_off=args.dist_cut_off,
                                metrics_cd=cd_results, metrics_iou=iou_results, metrics_nc=nc_results)


def main(argv=None):
    args = parse_arguments(argv)
    comparison_rec_mesh_template(args=args)


if __name__ == '__main__':
    main()
