import os
import sys
sys.path.append(os.path.abspath('.'))

from source import make_comparison

if __name__ == '__main__':

    workers = 15  # for training PC
    # workers = 8  # for Windows missing fork
    # workers = 4  # for strange window size bug
    # workers = 0  # debug

    comp_name = 'comp_ablation_abc_varnoise'
    dataset = 'abc'

    methods = [
        'ppsurf_vanilla',
        'ppsurf_vanilla_zeros_global',
        'ppsurf_vanilla_zeros_local',
        'ppsurf_vanilla_qpoints',
        'ppsurf_sym_max',
        'ppsurf_10nn',
        'ppsurf_25nn',
        'ppsurf_50nn',
        'ppsurf_merge_sum',
        'ppsurf_200nn',
    ]

    params = [
        '--comp_name', dataset,
        '--comp_dir', 'results/comp',
        '--comp_mean_name', comp_name,
        '--html_name', comp_name,
        '--data_dir', 'datasets/' + dataset,
        '--testset', 'testset.txt',
        '--results_dir', 'results',

        '--workers', str(workers),
        '--dist_cut_off', str(0.01),

        '--result_headers', *methods,
        '--result_paths', *[r'results/{}/'.format(m) + dataset for m in methods],
    ]
    make_comparison.main(argv=params)

    # Convert xlsx to latex
    from source.base.evaluation import xslx_to_latex
    ablation_xlsx = os.path.join('results', 'comp', dataset, comp_name + '.xlsx')
    xslx_to_latex(ablation_xlsx, ablation_xlsx[:-5] + '.tex')
