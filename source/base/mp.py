import subprocess
import multiprocessing
import typing
import os


def mp_worker(call):
    """
    Small function that starts a new thread with a system call. Used for thread pooling.
    :param call:
    :return:
    """
    call = call.split(' ')
    verbose = call[-1] == '--verbose'
    if verbose:
        call = call[:-1]
        subprocess.run(call)
    else:
        # subprocess.run(call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # suppress outputs
        subprocess.run(call, stdout=subprocess.DEVNULL)


def start_process_pool(worker_function, parameters: typing.Iterable[typing.Iterable], num_processes, timeout=None):
    from tqdm import tqdm

    if len(parameters) > 0:
        if num_processes <= 1:
            print('Running loop for {} with {} calls on {} workers'.format(
                str(worker_function), len(parameters), num_processes))
            results = []
            for c in tqdm(parameters):
                results.append(worker_function(*c))
            return results
        else:
            print('Running loop for {} with {} calls on {} subprocess workers'.format(
                str(worker_function), len(parameters), num_processes))

            results = []
            context = multiprocessing.get_context('spawn')  # 2023-10-25 fork got stuck on Linux (Python 3.9.12)
            pool = context.Pool(processes=num_processes, maxtasksperchild=1)
            try:
                # quick and easy TQDM, a bit laggy but works
                for result in pool.starmap(worker_function, tqdm(parameters, total=len(parameters))):
                # for result in pool.starmap(worker_function, parameters):  # without TQDM
                    results.append(result)
            except KeyboardInterrupt:
                # Allow ^C to interrupt from any thread.
                exit()
            pool.close()
            return results
    else:
        return None


def start_thread(func, args: typing.Sequence, kwargs={}):
    import threading

    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.start()
    return thread


def start_process(func, args: typing.Sequence, start_method=None):
    import multiprocessing as mp
    if start_method is None:
        proc = mp.Process(target=func, args=args)
    else:
        ctx = mp.get_context(start_method)
        proc = ctx.Process(target=func, args=args)
    proc.start()
    return proc


def get_multi_gpu_params(max_workers: typing.Optional[int] = None) -> typing.List[str]:
    from torch.cuda import device_count

    num_gpus = device_count()
    if num_gpus <= 1:
        return []

    num_workers = os.cpu_count()
    if max_workers is not None:
        num_workers = min(num_workers, max_workers)

    res_args = [
        '--trainer.strategy', 'ddp',
        # '--trainer.strategy', 'ddp_find_unused_parameters_true',  # for debugging
        '--model.init_args.workers', str(num_workers),
        '--data.init_args.use_ddp', str(True),
        '--data.init_args.workers', str(num_workers),
        '--data.init_args.batch_size', str(50 // num_gpus),
    ]

    return res_args
