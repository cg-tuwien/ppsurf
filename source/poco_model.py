import typing
import os

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from source.base.nn import FKAConvNetwork, batch_gather, count_parameters
from source.base import fs
from source.base.metrics import compare_predictions_binary_tensors

from source.poco_data_loader import get_proj_ids, get_data_poco


# Adapted from POCO: https://github.com/valeoai/POCO
# which is published under Apache 2.0: https://github.com/valeoai/POCO/blob/main/LICENSE

class PocoModel(pl.LightningModule):

    def __init__(self, output_names, in_channels, out_channels, k,
                 lambda_l1, debug, in_file, results_dir, padding_factor, name, network_latent_size,
                 gen_subsample_manifold_iter, gen_subsample_manifold, gen_resolution_global,
                 rec_batch_size, gen_refine_iter, workers):
        super().__init__()

        self.output_names = output_names
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.lambda_l1 = lambda_l1
        self.network_latent_size = network_latent_size
        self.gen_subsample_manifold_iter = gen_subsample_manifold_iter
        self.gen_subsample_manifold = gen_subsample_manifold
        self.gen_resolution_global = gen_resolution_global
        self.gen_resolution_metric = None
        self.num_pts_local = None
        self.rec_batch_size = rec_batch_size
        self.gen_refine_iter = gen_refine_iter
        self.workers = workers

        self.in_file = in_file
        self.results_dir = results_dir
        self.padding_factor = padding_factor

        self.debug = debug
        self.show_unused_params = debug
        self.name = name

        self.network = PocoNetwork(in_channels=self.in_channels, latent_size=self.network_latent_size,
                                   out_channels=self.out_channels, k=self.k)

        self.test_step_outputs = []
        
    def on_after_backward(self):  
        # for finding disconnected parts
        # DDP won't run by default if such parameters exist 
        # find_unused_parameters makes it run but is slower
        if self.show_unused_params:
            for name, param in self.named_parameters():
                if param.grad is None:
                    print('Unused param {}'.format(name))
            self.show_unused_params = False  # print once is enough

    def get_prog_bar(self):
        from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
        prog_bar = self.trainer.progress_bar_callback
        if prog_bar is not None and not isinstance(prog_bar, TQDMProgressBar):
            print('Warning: invalid progress bar type: {}'.format(type(prog_bar)))
        else:
            prog_bar = typing.cast(typing.Optional[TQDMProgressBar], prog_bar)
        return prog_bar

    def compute_loss(self, pred, batch_data):
        loss_components = []

        occ_target = batch_data['occ']
        occ_loss = nn.functional.cross_entropy(input=pred, target=occ_target, reduction='none')
        loss_components.append(occ_loss)

        loss_components_mean = [torch.mean(l) for l in loss_components]

        loss_components = torch.stack(loss_components)
        loss_components_mean = torch.stack(loss_components_mean)
        loss_tensor = loss_components_mean.mean()

        return loss_tensor, loss_components_mean, loss_components

    def calc_metrics(self, pred, gt_data):

        def compare_classification(pred, gt):
            pred_labels = torch.argmax(pred, dim=1).to(torch.float32)

            eval_dict = compare_predictions_binary_tensors(
                ground_truth=gt.squeeze(), predicted=pred_labels.squeeze(), prediction_name=None)
            return eval_dict

        eval_dict = compare_classification(pred=pred, gt=gt_data['occ'])
        eval_dict['abs_dist_rms'] = np.nan
        return eval_dict

    def get_loss_and_metrics(self, pred, batch):
        loss, loss_components_mean, loss_components = self.compute_loss(pred=pred, batch_data=batch)
        metrics_dict = self.calc_metrics(pred=pred, gt_data=batch)
        return loss, loss_components_mean, loss_components, metrics_dict

    def default_step_dict(self, batch):
        pred = self.network.forward(batch)
        loss, loss_components_mean, loss_components, metrics_dict = self.get_loss_and_metrics(pred, batch)

        if self.lambda_l1 != 0.0:
            loss = self.regularize(loss)

        if bool(self.debug):
            self.visualize_step_results(batch_data=batch, predictions=pred,
                                        losses=loss_components, metrics=metrics_dict)
        return loss, loss_components_mean, loss_components, metrics_dict

    def training_step(self, batch, batch_idx):
        loss, loss_components_mean, loss_components, metrics_dict = self.default_step_dict(batch=batch)
        self.do_logging(loss, loss_components_mean, log_type='train',
                        output_names=self.output_names, metrics_dict=metrics_dict, f1_in_prog_bar=False,
                        keys_to_log=frozenset({'accuracy', 'precision', 'recall', 'f1_score'}))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_components_mean, loss_components, metrics_dict = self.default_step_dict(batch=batch)
        self.do_logging(loss, loss_components_mean, log_type='val',
                        output_names=self.output_names, metrics_dict=metrics_dict, f1_in_prog_bar=True,
                        keys_to_log=frozenset({'accuracy', 'precision', 'recall', 'f1_score'}))
        return loss

    def test_step(self, batch, batch_idx):
        pred = self.network.forward(batch)

        # assume batch size is 1
        if batch['shape_id'].shape[0] != 1:
            raise NotImplementedError('batch size > 1 not supported')

        shape_id = batch['shape_id']

        loss, loss_components_mean, loss_components = self.compute_loss(pred=pred, batch_data=batch)
        metrics_dict = self.calc_metrics(pred=pred, gt_data=batch)

        if bool(self.debug):
            self.visualize_step_results(batch_data=batch, predictions=pred,
                                        losses=loss_components, metrics=metrics_dict)

        shape_id = shape_id.squeeze(0)
        loss_components_mean = loss_components_mean.squeeze(0)
        loss_components = loss_components.squeeze(0)
        pc_file_in = batch['pc_file_in'][0]

        results = {'shape_id': shape_id, 'pc_file_in': pc_file_in, 'loss': loss,
                   'loss_components_mean': loss_components_mean,
                   'loss_components': loss_components, 'metrics_dict': metrics_dict}
        self.test_step_outputs.append(results)

        prog_bar = self.get_prog_bar()
        prog_bar.test_progress_bar.set_postfix_str('pc_file: {}'.format(os.path.basename(pc_file_in)), refresh=True)
        return results

    def on_test_epoch_end(self):

        from source.base.evaluation import make_test_report
        from source.base.container import aggregate_dicts, flatten_dicts
        from source.occupancy_data_module import read_shape_list, get_results_dir

        shape_names = read_shape_list(self.in_file)
        results_dir = get_results_dir(out_dir=self.results_dir, name=self.name, in_file=self.in_file)

        outputs_flat = flatten_dicts(self.test_step_outputs)
        metrics_dicts_stacked = aggregate_dicts(outputs_flat, method='stack')

        output_file = os.path.join(results_dir, 'metrics_{}.xlsx'.format(self.name))
        loss_total_mean, abs_dist_rms_mean, f1_mean = make_test_report(
            shape_names=shape_names, results=metrics_dicts_stacked,
            output_file=output_file, output_names=self.output_names, is_dict=True)

        print('Test results (mean): Loss={}, RMSE={}, F1={}'.format(loss_total_mean, abs_dist_rms_mean, f1_mean))

    def predict_step(self, batch: dict, batch_idx, dataloader_idx=0):
        from source.occupancy_data_module import get_results_dir, in_file_is_dataset

        shape_data_poco = get_data_poco(batch_data=batch)
        prog_bar = self.get_prog_bar()

        if batch['pts_ms'].shape[0] > 1:
            raise NotImplementedError('batch size > 1 not supported')

        pc_file_in = batch['pc_file_in'][0]
        if in_file_is_dataset(self.in_file):
            results_dir = get_results_dir(out_dir=self.results_dir, name=self.name, in_file=self.in_file)
            out_file_rec = os.path.join(results_dir, 'meshes', os.path.basename(pc_file_in))
        else:
            # simple folder structure for single reconstruction
            out_file_basename = os.path.basename(pc_file_in) + '.ply'
            out_file_rec = os.path.join(self.results_dir, os.path.basename(pc_file_in), out_file_basename)
        pts = shape_data_poco['pts'][0].transpose(0, 1)

        # create the latent storage
        latent = torch.zeros((pts.shape[0], self.network_latent_size),
                             dtype=torch.float, device=pts.device)
        counts = torch.zeros((pts.shape[0],), dtype=torch.float, device=pts.device)

        iteration = 0
        for current_value in range(self.gen_subsample_manifold_iter):
            while counts.min() < current_value + 1:
                valid_ids = torch.argwhere(counts == current_value)[:, 0].clone().detach().long()

                if pts.shape[0] >= self.gen_subsample_manifold:

                    ids = torch.randperm(valid_ids.shape[0])[:self.gen_subsample_manifold]
                    ids = valid_ids[ids]

                    if ids.shape[0] < self.gen_subsample_manifold:
                        ids = torch.cat(
                            [ids, torch.randperm(pts.shape[0], device=pts.device)[
                                  :self.gen_subsample_manifold - ids.shape[0]]],
                            dim=0)
                    assert (ids.shape[0] == self.gen_subsample_manifold)
                else:
                    ids = torch.arange(pts.shape[0])

                data_partial = {'pts': shape_data_poco['pts'][0].transpose(1, 0)[ids].transpose(1, 0).unsqueeze(0)}
                partial_latent = self.network.get_latent(data_partial)['latents']
                latent[ids] += partial_latent[0].transpose(1, 0)
                counts[ids] += 1

                iteration += 1
                prog_bar.predict_progress_bar.set_postfix_str('get_latent iter: {}'.format(iteration), refresh=True)

        latent = latent / counts.unsqueeze(1)
        latent = latent.transpose(1, 0).unsqueeze(0)
        shape_data_poco['latents'] = latent
        latent = shape_data_poco

        from source.poco_utils import export_mesh_and_refine_vertices_region_growing_v3
        mesh = export_mesh_and_refine_vertices_region_growing_v3(
            network=self.network, latent=latent,
            pts_raw_ms=batch['pts_raw_ms'] if 'pts_raw_ms' in batch.keys() else None,
            resolution=self.gen_resolution_global,
            padding=1,
            mc_value=0,
            num_pts=self.rec_batch_size,
            num_pts_local=self.num_pts_local,
            input_points=shape_data_poco['pts'][0].cpu().numpy().transpose(1, 0),
            refine_iter=self.gen_refine_iter,
            out_value=1,
            prog_bar=prog_bar,
            pc_file_in=pc_file_in,
            # workers=self.workers,
        )

        if mesh is not None:
            # de-normalize if not part of a dataset
            from source.occupancy_data_module import in_file_is_dataset
            if not in_file_is_dataset(pc_file_in):
                from source.base.math import get_points_normalization_info, denormalize_points_with_info
                from source.occupancy_data_module import OccupancyDataModule
                pts_np = OccupancyDataModule.load_pts(pts_file=pc_file_in)
                pts_np, _ = OccupancyDataModule.pre_process_pts(pts=pts_np)
                bb_center, scale = get_points_normalization_info(pts=pts_np, padding_factor=self.padding_factor)
                mesh.vertices = denormalize_points_with_info(pts=mesh.vertices, bb_center=bb_center, scale=scale)

            # print(out_file_rec)
            fs.make_dir_for_file(out_file_rec)
            mesh.export(file_obj=out_file_rec)
        else:
            print('No reconstruction for {}'.format(pc_file_in))

        return 0  # return something to suppress warning

    def on_predict_epoch_end(self):
        from source.base.profiling import get_now_str
        from source.occupancy_data_module import get_results_dir, read_shape_list, get_meshes_dir, in_file_is_dataset

        if not in_file_is_dataset(self.in_file):
            return  # no dataset -> nothing to evaluate

        print('{}: Evaluating {}'.format(get_now_str(), self.name))
        from source.base import evaluation, fs

        results_dir = get_results_dir(out_dir=self.results_dir, name=self.name, in_file=self.in_file)
        shape_names = read_shape_list(self.in_file)
        gt_meshes_dir = get_meshes_dir(in_file=self.in_file)
        if not os.path.exists(gt_meshes_dir):
            print('Warning: {} not found. Skipping evaluation.'.format(gt_meshes_dir))
        else:
            gt_meshes = [os.path.join(gt_meshes_dir, '{}.ply'.format(vs)) for vs in shape_names]
            os.makedirs(results_dir, exist_ok=True)
            result_headers = [self.name]
            result_file_templates = [os.path.join(results_dir, 'meshes/{}.ply')]
            _ = evaluation.make_quantitative_comparison(
                shape_names=shape_names, gt_mesh_files=gt_meshes,
                result_headers=result_headers, result_file_templates=result_file_templates,
                comp_output_dir=results_dir, num_processes=self.workers, num_samples=100000)

        print('{}: Evaluating {} finished'.format(get_now_str(), self.name))

    def do_logging(self, loss_total, loss_components, log_type: str, output_names: list, metrics_dict: dict,
                   keys_to_log=frozenset({'abs_dist_rms', 'accuracy', 'precision', 'recall', 'f1_score'}),
                   f1_in_prog_bar=True, on_step=True, on_epoch=False):

        import math
        import numbers

        self.log('loss/{}/00_all'.format(log_type), loss_total, on_step=on_step, on_epoch=on_epoch)
        if len(loss_components) > 1:
            for li, l in enumerate(loss_components):
                self.log('loss/{}/{}_{}'.format(log_type, li, output_names[li]), l, on_step=on_step, on_epoch=on_epoch)

        for key in metrics_dict.keys():
            if key in keys_to_log and isinstance(metrics_dict[key], numbers.Number):
                value = metrics_dict[key]
                if math.isnan(value):
                    value = 0.0
                self.log('metrics/{}/{}'.format(log_type, key), value, on_step=on_step, on_epoch=on_epoch)

        self.log('metrics/{}/{}'.format(log_type, 'F1'), metrics_dict['f1_score'],
                 on_step=on_step, on_epoch=on_epoch, logger=False, prog_bar=f1_in_prog_bar)

    def visualize_step_results(self, batch_data: dict, predictions, losses, metrics):
        from source.base import visualization
        query_pts_ms = batch_data['pts_query_ms'].detach().cpu().numpy()
        occ_loss = losses[0].detach().cpu().numpy()
        vis_to_eval_file = os.path.join('debug', 'occ_loss_vis', 'test' + '.ply')
        visualization.plot_pts_scalar_data(query_pts_ms, occ_loss, vis_to_eval_file, prop_min=0.0, prop_max=1.0)


class PocoNetwork(pl.LightningModule):

    def __init__(self, in_channels, latent_size, out_channels, k):
        super().__init__()

        self.encoder = FKAConvNetwork(in_channels, latent_size, segmentation=True, dropout=0, x4d_bug_fixed=False)
        self.projection = InterpAttentionKHeadsNet(latent_size, out_channels, k)

        self.lcp_preprocess = True

        print(f'Network -- backbone -- {count_parameters(self.encoder)} parameters')
        print(f'Network -- projection -- {count_parameters(self.projection)} parameters')

    def forward(self, data):
        latents = self.encoder.forward(data, spectral_only=True)
        data['latents'] = latents
        ret_data = self.projection.forward(data, has_proj_ids=True)
        return ret_data

    def get_latent(self, data):
        latents = self.encoder.forward(data, spectral_only=False)
        data['latents'] = latents
        data['proj_correction'] = None
        return data

    def from_latent(self, data: typing.Dict[str, torch.Tensor]):
        data_proj = self.projection.forward(data)
        return data_proj


class InterpAttentionKHeadsNet(torch.nn.Module):

    def __init__(self, latent_size, out_channels, k=16):
        super().__init__()

        print(f'InterpNet - Simple - K={k}')
        self.fc1 = torch.nn.Conv2d(latent_size + 3, latent_size, 1)
        self.fc2 = torch.nn.Conv2d(latent_size, latent_size, 1)
        self.fc3 = torch.nn.Conv2d(latent_size, latent_size, 1)

        self.fc8 = torch.nn.Conv1d(latent_size, out_channels, 1)

        self.fc_query = torch.nn.Conv2d(latent_size, 64, 1)
        self.fc_value = torch.nn.Conv2d(latent_size, latent_size, 1)

        self.k = k

        self.activation = torch.nn.ReLU()

    def forward(self, data: typing.Dict[str, torch.Tensor], has_proj_ids: bool = False, last_layer: bool = True)\
            -> torch.Tensor:

        if not has_proj_ids:
            spatial_data = get_proj_ids(data, self.k)
            for key, value in spatial_data.items():
                data[key] = value

        x = data['latents']
        indices = data['proj_ids']
        pts = data['pts']
        pts_query = data['pts_query'].to(pts.device)

        if pts.shape[1] != 3:
            pts = pts.transpose(1, 2)

        if pts_query.shape[1] != 3:
            pts_query = pts_query.transpose(1, 2)

        x = batch_gather(x, 2, indices)
        pts = batch_gather(pts, 2, indices)
        pts = pts_query.unsqueeze(3) - pts

        x = torch.cat([x, pts], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        query = self.fc_query(x)
        value = self.fc_value(x)

        attention = torch.nn.functional.softmax(query, dim=-1).mean(dim=1)
        x = torch.matmul(attention.unsqueeze(-2), value.permute(0, 2, 3, 1)).squeeze(-2)
        x = x.transpose(1, 2)

        if last_layer:
            x = self.fc8(x)

        return x
