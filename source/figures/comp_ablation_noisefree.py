import os
import sys
sys.path.append(os.path.abspath('.'))

from source.base.evaluation import merge_comps
from source import make_comparison

if __name__ == '__main__':
    workers = 15  # for training PC
    # workers = 8  # for Windows missing fork
    # workers = 4  # for strange window size bug
    # workers = 0  # debug

    comp_name = 'comp_ablation_noisefree'

    datasets = [
        'abc_noisefree',
        'famous_noisefree',
        'thingi10k_scans_noisefree',
    ]

    methods = [
        'ppsurf_25nn',
        'ppsurf_50nn',
        'ppsurf_vanilla',
        'ppsurf_merge_sum',
    ]

    # Run all comparisons
    for dataset in datasets:
        print('Running comparison for dataset {}'.format(dataset))
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
        try:
            make_comparison.main(argv=params)
        except Exception as e:
            print('Error in dataset {}: {}'.format(dataset, e))

    # Merge all comparisons
    comp_files = ['results/comp/{}/{}.xlsx'.format(dataset, comp_name) for dataset in datasets]
    comp_merged_xlsx = 'results/comp/reports/{}.xlsx'.format(comp_name)
    comp_merged_latex = 'results/comp/reports/{}.tex'.format(comp_name)
    merge_comps(comp_files, comp_merged_xlsx, comp_merged_latex, methods_order=methods, float_format='%.3f')
