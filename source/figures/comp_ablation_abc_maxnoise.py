import os
import sys
sys.path.append(os.path.abspath('.'))

from source import make_comparison

if __name__ == '__main__':

    workers = 15  # for training PC
    # workers = 8  # for Windows missing fork
    # workers = 4  # for strange window size bug
    # workers = 0  # debug

    comp_name = 'comp_ablation_abc_extra_noisy'
    # comp_name = 'comp_ablation_abc_varnoise'
    comp_dir = 'results/comp'

    dataset = 'abc_extra_noisy'

    params = [
        '--comp_name', comp_name,
        '--comp_dir', comp_dir,
        '--comp_mean_name', 'comp_mean',
        '--html_name', 'comp_all',
        '--data_dir', 'datasets/' + dataset,
        '--testset', 'testset.txt',
        '--results_dir', 'results',
        '--result_headers',
        'ppsurf_merge_cat',
        'ppsurf_vanilla_zeros_global',
        'ppsurf_vanilla_zeros_local',
        'ppsurf_vanilla_qpoints',
        'ppsurf_sym_max',
        'ppsurf_10nn',
        'p2s2_25nn',
        'p2s2_50nn',
        'p2s2_merge_sum',
        'p2s2_200nn',
        '--result_paths',
        r'results/p2s2_vanilla/' + dataset,
        r'results/p2s2_vanilla_zeros_global/' + dataset,
        r'results/p2s2_vanilla_zeros_local/' + dataset,
        r'results/p2s2_vanilla_qpoints/' + dataset,
        r'results/p2s2_sym_max/' + dataset,
        r'results/p2s2_10nn/' + dataset,
        r'results/p2s2_25nn/' + dataset,
        r'results/p2s2_50nn/' + dataset,
        r'results/p2s2_merge_sum/' + dataset,
        r'results/p2s2_200nn/' + dataset,

        '--workers', str(workers),
        '--dist_cut_off', str(0.01),
    ]
    make_comparison.main(argv=params)

    # Convert xlsx to latex
    from source.base.evaluation import xslx_to_latex
    ablation_xlsx = os.path.join(comp_dir, comp_name, 'comp_mean.xlsx')
    xslx_to_latex(ablation_xlsx, ablation_xlsx[:-5] + '.tex')

    print('Points2Surf2 Comparison is finished!')
