# this script runs a model through training, testing and prediction of all datasets

import os
from source.base.mp import get_multi_gpu_params

if __name__ == '__main__':
    python_call = 'python'
    main_cmd = 'pps.py'
    name = 'ppsurf_50nn'
    version = '0'
    # on_server = True

    main_cmd = python_call + ' ' + main_cmd

    cmd_template = '{main_cmd} {sub_cmd} {configs}'
    configs = '-c configs/poco.yaml -c configs/ppsurf.yaml {server} -c configs/{name}.yaml'

    # training
    # configs_train = configs.format(server='-c configs/device_server.yaml' if on_server else '', name=name)
    configs_train = configs.format(server=' '.join(get_multi_gpu_params()), name=name)
    cmd_train = cmd_template.format(main_cmd=main_cmd, sub_cmd='fit', configs=configs_train)
    os.system(cmd_train)

    args_no_train = ('--ckpt_path models/{name}/version_{version}/checkpoints/last.ckpt '
                     '--trainer.logger False --trainer.devices 1').format(name=name, version=version)
    configs_no_train = configs.format(server='', name=name)
    cmd_template_no_train = cmd_template + ' --data.init_args.in_file {dataset}/testset.txt ' + args_no_train

    # testing
    cmd_test = cmd_template_no_train.format(main_cmd=main_cmd, sub_cmd='test', configs=configs_no_train,
                                            dataset='datasets/abc_train')
    os.system(cmd_test)

    # prediction
    datasets = [
        # 'abc_minimal',
        'abc',
        'abc_extra_noisy',
        'abc_noisefree',
        'real_world',
        'famous_original', 'famous_noisefree', 'famous_sparse', 'famous_dense', 'famous_extra_noisy',
        'thingi10k_scans_original', 'thingi10k_scans_noisefree', 'thingi10k_scans_sparse',
        'thingi10k_scans_dense', 'thingi10k_scans_extra_noisy'
        ]
    for ds in datasets:
        cmd_pred = cmd_template_no_train.format(main_cmd=main_cmd, sub_cmd='predict', configs=configs_no_train,
                                                dataset='datasets/' + ds)
        os.system(cmd_pred)

    # make comparison
    os.system('python source/figures/comp_all.py')

    print('All done. You should find the results in results/comp/reports/comp_all.xlsx.')
