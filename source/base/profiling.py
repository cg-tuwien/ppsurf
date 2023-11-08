import tracemalloc
import linecache
import typing


def init_profiling():
    tracemalloc.start()


def compare_snaps(snap1, snap2, limit=50):
    top_stats = snap1.compare_to(snap2, 'lineno')

    for stat in top_stats[:limit]:
        line = str(stat)
        # if '~/' in line:  # filter only lines from my own code
        print(line)


def display_top(snapshot: typing.Union[tracemalloc.Snapshot, None], key_type='lineno', limit=10):
    if snapshot is None:
        snapshot = tracemalloc.take_snapshot()

    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, '<frozen importlib._bootstrap>'),
        tracemalloc.Filter(False, '<unknown>'),
    ))
    top_stats = snapshot.statistics(key_type)

    print('Top %s lines' % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print('#%s: %s:%s: %.1f KiB'
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print('%s other: %.1f KiB' % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print('Total allocated size: %.1f KiB' % (total / 1024))


def print_duration(func, params: dict, name: str):
    import time
    start = time.time()
    res = func(**params)
    end = time.time()
    print('{} took: {}'.format(name, end - start))
    return res


def print_memory(min_num_bytes=0):
    import sys
    import gc

    objects = gc.get_objects()

    objects_sizes = dict()
    for obj_id, obj in enumerate(objects):
        num_bytes = sys.getsizeof(obj)
        if num_bytes >= min_num_bytes:
            name = str(type(obj)) + str(obj_id)
            objects_sizes[name] = num_bytes

    objects_sizes_sorted = dict(sorted(objects_sizes.items(), key=lambda item: item[1], reverse=True))
    print('Objects in scope:')
    for name, num_bytes in objects_sizes_sorted.items():
        print('{}: {} kB'.format(name, num_bytes / 1024))
    print('')


def get_now_str():
    import datetime
    return str(datetime.datetime.now())
