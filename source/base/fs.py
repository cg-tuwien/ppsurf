import typing
import os


def create_activate_env(env_name: str):
    import subprocess

    # check if conda is installed
    def _check_conda_installed(command: str):
        try:
            conda_output = subprocess.check_output([command, '-V']).decode('utf-8').strip()
        except Exception as _:
            return False
        conda_output_regex = r'conda (\d+\.\d+\.\d+)'  # mamba and conda both output 'conda 23.3.1'
        import re
        conda_match = re.match(conda_output_regex, conda_output)
        return conda_match is not None

    if _check_conda_installed('mamba'):
        conda_exe = 'mamba'
    elif _check_conda_installed('conda'):
        conda_exe = 'conda'
    else:
        raise ValueError('Conda not found')

    # check if env already exists. example outputs:
    # conda env list --json
    # {
    #   "envs": [
    #     "C:\\miniforge",
    #     "C:\\miniforge\\envs\\pps"
    #   ]
    # }
    import json
    env_list_str = subprocess.check_output([conda_exe, 'env', 'list', '--json']).decode('utf-8')
    env_list_json = json.loads(env_list_str)
    envs = env_list_json['envs']
    envs_dirs = [os.path.split(env)[1] for env in envs]
    first_run = env_name not in envs_dirs
    if first_run:
        import subprocess
        on_windows = os.name == 'nt'
        yml_file = '{}{}.yml'.format(env_name, '_win' if on_windows else '')
        env_install_cmd = [conda_exe, 'env', 'create', '--file', yml_file]
        print('Creating conda environment from {}\n{}'.format(yml_file, env_install_cmd))
        subprocess.call(env_install_cmd)

    # conda activate pps
    subprocess.call([conda_exe, 'activate', env_name])

    if first_run:
        print('Downloading datasets')
        subprocess.call(['python', 'source/datasets/download_abc_training.py'])
        subprocess.call(['python', 'source/datasets/download_testsets.py'])


def make_dir_for_file(file):
    file_dir = os.path.dirname(file)
    if file_dir != '':
        if not os.path.exists(file_dir):
            try:
                os.makedirs(os.path.dirname(file), exist_ok=True)
            except FileExistsError as exc:
                pass
            except OSError as exc:  # Guard against race condition
                raise


def call_necessary(file_in: typing.Union[str, typing.Sequence[str]], file_out: typing.Union[str, typing.Sequence[str]],
                   min_file_size=0, verbose=False):
    """
    Check if all input files exist and at least one output file does not exist or is invalid.
    :param file_in: list of str or str
    :param file_out: list of str or str
    :param min_file_size: int
    :return:
    """

    def check_parameter_types(param):
        if isinstance(param, str):
            return [param]
        elif isinstance(param, list):
            return param
        elif isinstance(param, tuple):
            return param
        else:
            raise ValueError('Wrong input type')

    file_in = check_parameter_types(file_in)
    file_out = check_parameter_types(file_out)

    def print_result(msg: str):
        if verbose:
            print('call_necessary\n {}\n ->\n {}: \n{}'.format(file_in, file_out, msg))

    if len(file_out) == 0:
        print_result('No output')
        return True

    inputs_missing = [f for f in file_in if not os.path.isfile(f)]
    if len(inputs_missing) > 0:
        print_result('WARNING: Input files are missing: {}'.format(inputs_missing))
        return False

    outputs_missing = [f for f in file_out if not os.path.isfile(f)]
    if len(outputs_missing) > 0:
        print_result('Some output files are missing: {}'.format(outputs_missing))
        return True

    min_output_file_size = min([os.path.getsize(f) for f in file_out])
    if min_output_file_size < min_file_size:
        print_result('Output too small')
        return True

    oldest_input_file_mtime = max([os.path.getmtime(f) for f in file_in])
    youngest_output_file_mtime = min([os.path.getmtime(f) for f in file_out])
    if oldest_input_file_mtime >= youngest_output_file_mtime:
        if verbose:
            import time
            import numpy as np
            input_file_mtime_arg_max = np.argmax(np.array([os.path.getmtime(f) for f in file_in]))
            output_file_mtime_arg_min = np.argmin(np.array([os.path.getmtime(f) for f in file_out]))
            input_file_mtime_max = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(oldest_input_file_mtime))
            output_file_mtime_min = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(youngest_output_file_mtime))
            print_result('Input file {} is newer than output file {}: {} >= {}'.format(
                file_in[input_file_mtime_arg_max], file_out[output_file_mtime_arg_min],
                input_file_mtime_max, output_file_mtime_min))
        return True

    return False


def text_file_lf_to_crlf(file):
    """
    Convert line endings of a text file from LF to CRLF.
    :param file:
    :return:
    """

    with open(file, 'r') as fp:
        lines = fp.readlines()

    with open(file, 'w') as fp:
        for line in lines:
            fp.write(line.rstrip() + '\r\n')
