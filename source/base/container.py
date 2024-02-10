import typing

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def flatten_dicts(dicts: typing.Sequence[typing.Dict[typing.Any, typing.Any]]):
    """
    flatten dicts containing other dicts
    :param dicts:
    :return:
    """
    if len(dicts) == 0:
        return dict()
    elif len(dicts) == 1:
        return dicts[0]

    new_dicts = []
    for d in dicts:
        new_dict = {}
        for k in d.keys():
            value = d[k]
            if isinstance(value, typing.Dict):
                new_dict.update(value)
            else:
                new_dict[k] = value
        new_dicts.append(new_dict)

    return new_dicts


def aggregate_dicts_np(dicts: typing.Sequence[typing.Mapping[typing.Any, np.ndarray]], method: str):
    """

    :param dicts:
    :param method: one of ['mean', 'concat', 'stack']
    :return:
    """
    import numbers

    if len(dicts) == 0:
        return dict()
    elif len(dicts) == 1 and method != 'stack':  # add singleton dimension for stack
        return dicts[0]

    valid_methods = ['mean', 'concat', 'stack']
    if method not in valid_methods:
        raise ValueError('Invalid method {} must be one of {}'.format(method, valid_methods))

    dict_aggregated = dict()
    for k in dicts[0].keys():
        values = [d[k] for d in dicts]
        if isinstance(values[0], numbers.Number):
            values = [np.asarray(v) for v in values]

        if method == 'concat':
            values_np = np.concatenate(values)
        elif method == 'stack':
            values_np = np.stack(values)
        elif method == 'mean':
            values_np = np.array(values)
            if values_np.dtype.type is np.str_:
                values_np = values_np[0]
            else:
                values_np = np.nanmean(values_np)
        else:
            raise ValueError()
        dict_aggregated[k] = values_np
    return dict_aggregated


def aggregate_dicts(dicts: typing.Sequence[typing.Mapping[typing.Any, 'torch.Tensor']], method: str):
    """

    :param dicts:
    :param method: one of ['mean', 'concat', 'stack']
    :return:
    """
    import torch
    import numbers

    if len(dicts) == 0:
        return dict()
    elif len(dicts) == 1:
        return dicts[0]

    valid_methods = ['mean', 'concat', 'stack']
    if method not in valid_methods:
        raise ValueError('Invalid method {} must be one of {}'.format(method, valid_methods))

    dict_aggregated = dict()
    for k in dicts[0].keys():
        values = [d[k] for d in dicts]
        if isinstance(values[0], numbers.Number):
            values = [torch.as_tensor(v) for v in values]

        if method == 'concat':
            values = torch.cat(values)
        elif method == 'stack':
            if isinstance(values[0], str):
                pass  # keep list of strings
            else:
                values = torch.stack(values)
        elif method == 'mean':
            values = torch.tensor(values)
            if values.dtype == str:
                values = values[0]
            else:
                values = torch.nanmean(values).item()
        else:
            raise ValueError()
        dict_aggregated[k] = values
    return dict_aggregated


def tensor_list_to_array(tensors: typing.Sequence['torch.Tensor']) -> np.ndarray:
    """
    use this if the tensors can be on different devices
    :param tensors:
    :return: array
    """
    import torch
    tensors_cpu = [t.detach().cpu() for t in tensors]
    arr = torch.concat(tensors_cpu).numpy()
    return arr


def dict_np_to_torch(patch_data: dict):
    # convert values to tensors if necessary
    from torch import from_numpy, Tensor, tensor
    import numbers

    for key in patch_data.keys():
        val = patch_data[key]
        if isinstance(val, np.ndarray):
            patch_data[key] = from_numpy(val.copy())
        elif isinstance(val, Tensor):
            pass  # nothing to do
        elif np.isscalar(val):
            if isinstance(val, numbers.Number):
                patch_data[key] = tensor(val)
            elif isinstance(val, str):
                patch_data[key] = val
            else:  # try to keep type and let pytorch handle it
                patch_data[key] = val
        else:
            raise NotImplementedError('Key: {}, Type:{}'.format(key, type(val)))
    return patch_data
