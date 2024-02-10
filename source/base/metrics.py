import os
import typing

import numpy as np

from source.base.point_cloud import sample_mesh
from source.base.proximity import kdtree_query_oneshot


def calc_accuracy(num_true, num_predictions):
    if num_predictions == 0:
        return float('NaN')
    else:
        return num_true / num_predictions


def calc_precision(num_true_pos, num_false_pos):
    if isinstance(num_true_pos, (int, float)) and isinstance(num_false_pos, (int, float)) and \
            num_true_pos + num_false_pos == 0:
        return float('NaN')
    else:
        return num_true_pos / (num_true_pos + num_false_pos)


def calc_recall(num_true_pos, num_false_neg):
    if isinstance(num_true_pos, (int, float)) and isinstance(num_false_neg, (int, float)) and \
            num_true_pos + num_false_neg == 0:
        return float('NaN')
    else:
        return num_true_pos / (num_true_pos + num_false_neg)


def calc_f1(precision, recall):
    if isinstance(precision, (int, float)) and isinstance(recall, (int, float)) and \
            precision + recall == 0:
        return float('NaN')
    else:
        return 2.0 * (precision * recall) / (precision + recall)


def compare_predictions_binary_tensors(ground_truth, predicted, prediction_name):
    """

    :param ground_truth:
    :param predicted:
    :param prediction_name:
    :return: res_dict, prec_per_patch
    """

    import torch

    if ground_truth.shape != predicted.shape:
        raise ValueError('The ground truth matrix and the predicted matrix have different sizes!')

    if not isinstance(ground_truth, torch.Tensor) or not isinstance(predicted, torch.Tensor):
        raise ValueError('Both matrices must be dense of type torch.tensor!')

    ground_truth_int = (ground_truth > 0.0).to(dtype=torch.int32)
    predicted_int = (predicted > 0.0).to(dtype=torch.int32)
    res_dict = dict()
    if prediction_name is not None:
        res_dict['comp_name'] = prediction_name

    res_dict['predictions'] = float(torch.numel(ground_truth_int))
    res_dict['pred_gt'] = float(torch.numel(ground_truth_int))
    res_dict['positives'] = float(torch.nonzero(predicted_int).shape[0])
    res_dict['pos_gt'] = float(torch.nonzero(ground_truth_int).shape[0])
    res_dict['true_neg'] = res_dict['predictions'] - float(torch.nonzero(predicted_int + ground_truth_int).shape[0])
    res_dict['negatives'] = res_dict['predictions'] - res_dict['positives']
    res_dict['neg_gt'] = res_dict['pred_gt'] - res_dict['pos_gt']
    true_pos = ((predicted_int + ground_truth_int) == 2).sum().to(dtype=torch.float32)
    res_dict['true_pos'] = float(true_pos.sum())
    res_dict['true'] = res_dict['true_pos'] + res_dict['true_neg']
    false_pos = ((predicted_int * 2 + ground_truth_int) == 2).sum().to(dtype=torch.float32)
    res_dict['false_pos'] = float(false_pos.sum())
    false_neg = ((predicted_int + 2 * ground_truth_int) == 2).sum().to(dtype=torch.float32)
    res_dict['false_neg'] = float(false_neg.sum())
    res_dict['false'] = res_dict['false_pos'] + res_dict['false_neg']
    res_dict['accuracy'] = calc_accuracy(res_dict['true'], res_dict['predictions'])
    res_dict['precision'] = calc_precision(res_dict['true_pos'], res_dict['false_pos'])
    res_dict['recall'] = calc_recall(res_dict['true_pos'], res_dict['false_neg'])
    res_dict['f1_score'] = calc_f1(res_dict['precision'], res_dict['recall'])

    return res_dict


def compare_predictions_binary_arrays(ground_truth: np.ndarray, predicted: np.ndarray, prediction_name):

    if ground_truth.shape != predicted.shape:
        raise ValueError('The ground truth matrix and the predicted matrix have different sizes!')

    ground_truth_int = (ground_truth > 0.0).astype(dtype=np.int32)
    predicted_int = (predicted > 0.0).astype(dtype=np.int32)
    res_dict = dict()
    res_dict['comp_name'] = prediction_name

    res_dict['predictions'] = float(ground_truth_int.size)
    res_dict['pred_gt'] = float(ground_truth_int.size)
    res_dict['positives'] = float(np.nonzero(predicted_int)[0].shape[0])
    res_dict['pos_gt'] = float(np.nonzero(ground_truth_int)[0].shape[0])
    res_dict['true_neg'] = res_dict['predictions'] - float(np.nonzero(predicted_int + ground_truth_int)[0].shape[0])
    res_dict['negatives'] = res_dict['predictions'] - res_dict['positives']
    res_dict['neg_gt'] = res_dict['pred_gt'] - res_dict['pos_gt']
    true_pos = ((predicted_int + ground_truth_int) == 2).sum().astype(dtype=np.float32)
    res_dict['true_pos'] = float(true_pos.sum())
    res_dict['true'] = res_dict['true_pos'] + res_dict['true_neg']
    false_pos = ((predicted_int * 2 + ground_truth_int) == 2).sum().astype(dtype=np.float32)
    res_dict['false_pos'] = float(false_pos.sum())
    false_neg = ((predicted_int + 2 * ground_truth_int) == 2).sum().astype(dtype=np.float32)
    res_dict['false_neg'] = float(false_neg.sum())
    res_dict['false'] = res_dict['false_pos'] + res_dict['false_neg']
    res_dict['accuracy'] = calc_accuracy(res_dict['true'], res_dict['predictions'])
    res_dict['precision'] = calc_precision(res_dict['true_pos'], res_dict['false_pos'])
    res_dict['recall'] = calc_recall(res_dict['true_pos'], res_dict['false_neg'])
    res_dict['f1_score'] = calc_f1(res_dict['precision'], res_dict['recall'])

    return res_dict


def chamfer_distance(file_in, file_ref, samples_per_model, num_processes=1):
    # http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf

    new_mesh_samples = sample_mesh(file_in, samples_per_model, rejection_radius=0.0)
    ref_mesh_samples = sample_mesh(file_ref, samples_per_model, rejection_radius=0.0)

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, -1.0

    ref_new_dist, corr_new_ids = kdtree_query_oneshot(pts=new_mesh_samples, pts_query=ref_mesh_samples,
                                                      k=1, sqr_dists=False)
    new_ref_dist, corr_ref_ids = kdtree_query_oneshot(pts=ref_mesh_samples, pts_query=new_mesh_samples,
                                                      k=1, sqr_dists=False)

    ref_new_dist_sum = np.sum(ref_new_dist)
    new_ref_dist_sum = np.sum(new_ref_dist)
    chamfer_dist = ref_new_dist_sum + new_ref_dist_sum
    chamfer_dist_mean = chamfer_dist / (new_mesh_samples.shape[0] + ref_mesh_samples.shape[0])

    return file_in, file_ref, chamfer_dist_mean


def hausdorff_distance(file_in, file_ref, samples_per_model):
    import scipy.spatial as spatial

    new_mesh_samples = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples = sample_mesh(file_ref, samples_per_model)

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, -1.0, -1.0, -1.0

    dist_new_ref, _, _ = spatial.distance.directed_hausdorff(new_mesh_samples, ref_mesh_samples)
    dist_ref_new, _, _ = spatial.distance.directed_hausdorff(ref_mesh_samples, new_mesh_samples)
    dist = max(dist_new_ref, dist_ref_new)
    return file_in, file_ref, dist_new_ref, dist_ref_new, dist


def intersection_over_union(file_in, file_ref, num_samples, num_dims=3):
    # https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/

    import trimesh
    from source.base.proximity import get_signed_distance_pysdf_inaccurate

    rng = np.random.default_rng(seed=42)
    samples = rng.random(size=(num_samples, num_dims)) - 0.5

    try:
        mesh_in = trimesh.load(file_in)
        mesh_ref = trimesh.load(file_ref)
    except:
        return file_in, file_ref, np.nan

    sdf_in = get_signed_distance_pysdf_inaccurate(mesh_in, samples)
    sdf_ref = get_signed_distance_pysdf_inaccurate(mesh_ref, samples)

    occ_in = sdf_in > 0.0
    occ_ref = sdf_ref > 0.0

    intersection = np.logical_and(occ_in, occ_ref)
    union = np.logical_or(occ_in, occ_ref)
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    if union_sum == 0.0:
        iou = 0.0
    else:
        iou = intersection_sum / union_sum

    return file_in, file_ref, iou


def f1_approx(file_in, file_ref, num_samples, num_dims=3):
    # https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/

    import trimesh
    from source.base.proximity import get_signed_distance_pysdf_inaccurate

    rng = np.random.default_rng(seed=42)
    samples = rng.random(size=(num_samples, num_dims)) - 0.5

    try:
        mesh_in = trimesh.load(file_in)
        mesh_ref = trimesh.load(file_ref)
    except:
        return file_in, file_ref, np.nan

    sdf_in = get_signed_distance_pysdf_inaccurate(mesh_in, samples)
    sdf_ref = get_signed_distance_pysdf_inaccurate(mesh_ref, samples)

    occ_in = sdf_in > 0.0
    occ_ref = sdf_ref > 0.0

    stats = compare_predictions_binary_arrays(occ_ref, occ_in, prediction_name='f1_approx')

    if np.isnan(stats['f1_score']):
        f1 = 0.0
    else:
        f1 = stats['f1_score']

    return file_in, file_ref, f1


def normal_error(file_in, file_ref, num_samples):

    import trimesh.sample
    from source.base import proximity

    try:
        mesh_in = trimesh.load(file_in)
        mesh_ref = trimesh.load(file_ref)
    except:
        return file_in, file_ref, np.nan

    samples, face_index = trimesh.sample.sample_surface(mesh_ref, num_samples)
    face_normals_ref = mesh_ref.face_normals[face_index]

    closest_points_in, distance, faces_in = proximity.get_closest_point_on_mesh(mesh_in, samples)
    face_normals_in = mesh_in.face_normals[faces_in]

    cosine = np.einsum('ij,ij->i', face_normals_ref, face_normals_in)
    cosine = np.clip(cosine, -1, 1)
    normal_c = np.nanmean(np.arccos(cosine))

    return file_in, file_ref, normal_c


def normal_error_approx(file_in, file_ref, num_samples=100000, num_processes=1):
    import trimesh.sample

    try:
        mesh_in = trimesh.load(file_in)
        mesh_ref = trimesh.load(file_ref)
    except:
        return file_in, file_ref, np.nan

    samples_rec, face_index_rec = trimesh.sample.sample_surface(mesh_in, num_samples)
    face_normals_rec = mesh_in.face_normals[face_index_rec]

    samples_gt, face_index_gt = trimesh.sample.sample_surface(mesh_ref, num_samples)
    face_normals_gt = mesh_ref.face_normals[face_index_gt]

    _, rec_ids = kdtree_query_oneshot(pts=samples_gt, pts_query=samples_rec, k=1, sqr_dists=True)

    face_normals_gt_nn = face_normals_gt[rec_ids]

    cosine = np.einsum('ij,ij->i', face_normals_rec, face_normals_gt_nn)
    cosine = np.clip(cosine, -1, 1)
    normal_c = np.nanmean(np.arccos(cosine))

    return file_in, file_ref, normal_c


def rmse(predictions: np.ndarray, targets: np.ndarray):
    return np.sqrt(((predictions - targets) ** 2).mean())


def get_metric_mesh_single_file(gt_mesh_file: str, mesh_file: str, num_samples: int,
                                metric: typing.Literal['chamfer', 'iou', 'normals', 'f1'] = 'chamfer') -> float:

    if os.path.isfile(mesh_file) and os.path.isfile(gt_mesh_file):
        if metric == 'chamfer':
            file_in, file_ref, metric_result = chamfer_distance(
                file_in=mesh_file, file_ref=gt_mesh_file, samples_per_model=num_samples)
        elif metric == 'iou':
            file_in, file_ref, metric_result = intersection_over_union(
                file_in=mesh_file, file_ref=gt_mesh_file, num_samples=num_samples)
        elif metric == 'normals':
            file_in, file_ref, metric_result = normal_error_approx(
                file_in=mesh_file, file_ref=gt_mesh_file, num_samples=num_samples)
        elif metric == 'f1':
            file_in, file_ref, metric_result = f1_approx(
                file_in=mesh_file, file_ref=gt_mesh_file, num_samples=num_samples)
        else:
            raise ValueError()
    elif not os.path.isfile(mesh_file):
        print('WARNING: mesh missing: {}'.format(mesh_file))
        metric_result = np.nan
        # raise FileExistsError()
    elif not os.path.isfile(gt_mesh_file):
        raise FileExistsError()
    else:
        raise NotImplementedError()

    return metric_result


def get_metric_meshes(result_file_template: typing.Sequence[str],
                      shape_list: typing.Sequence[str], gt_mesh_files: typing.Sequence[str],
                      num_samples=10000, metric: typing.Literal['chamfer', 'iou', 'normals', 'f1'] = 'chamfer',
                      num_processes=1) \
        -> typing.Iterable[np.ndarray]:
    from source.base.mp import start_process_pool

    metric_results = []
    for template in result_file_template:
        cd_params = []
        for sni, shape_name in enumerate(shape_list):
            gt_mesh_file = gt_mesh_files[sni]
            mesh_file = template.format(shape_name)
            cd_params.append((gt_mesh_file, mesh_file, num_samples, metric))

        metric_results.append(np.array(start_process_pool(
            worker_function=get_metric_mesh_single_file, parameters=cd_params, num_processes=num_processes)))

    return metric_results
