import typing

import pytorch_lightning as pl
import torch

from source.poco_model import PocoModel
from source.base.nn import count_parameters


class PPSurfModel(PocoModel):

    def __init__(self,
                 pointnet_latent_size,
                 output_names, in_channels, out_channels, k,
                 lambda_l1, debug, in_file, results_dir, padding_factor, name, network_latent_size,
                 gen_subsample_manifold_iter, gen_subsample_manifold, gen_resolution_global, num_pts_local,
                 rec_batch_size, gen_refine_iter, workers
                 ):
        super(PPSurfModel, self).__init__(
            output_names=output_names, in_channels=in_channels,
            out_channels=out_channels, k=k,
            lambda_l1=lambda_l1, debug=debug, in_file=in_file, results_dir=results_dir,
            padding_factor=padding_factor, name=name, workers=workers, rec_batch_size=rec_batch_size,
            gen_refine_iter=gen_refine_iter, gen_subsample_manifold=gen_subsample_manifold,
            gen_resolution_global=gen_resolution_global,
            gen_subsample_manifold_iter=gen_subsample_manifold_iter,
            network_latent_size=network_latent_size
        )

        self.num_pts_local = num_pts_local
        self.pointnet_latent_size = pointnet_latent_size

        self.network = PPSurfNetwork(in_channels=self.in_channels, latent_size=self.network_latent_size,
                                     out_channels=self.out_channels, k=self.k,
                                     num_pts_local=self.num_pts_local,
                                     pointnet_latent_size=self.pointnet_latent_size)


class PPSurfNetwork(pl.LightningModule):

    def __init__(self, in_channels, latent_size, out_channels, k, num_pts_local, pointnet_latent_size):
        super().__init__()

        from source.poco_model import InterpAttentionKHeadsNet
        from source.base.nn import FKAConvNetwork
        from source.base.nn import PointNetfeat, MLP

        self.latent_size = latent_size
        self.encoder = FKAConvNetwork(in_channels, latent_size, segmentation=True, dropout=0,
                                      activation=torch.nn.SiLU(), x4d_bug_fixed=True)
        self.projection = InterpAttentionKHeadsNet(latent_size, latent_size, k)
        self.point_net = PointNetfeat(net_size_max=pointnet_latent_size, num_points=num_pts_local, use_point_stn=False,
                                      use_feat_stn=True, output_size=latent_size, sym_op='att', dim=3)

        # self.branch_att = AttentionPoco(latent_size, reduce=True)  # attention ablation

        # self.mlp = MLP(input_size=latent_size*2, output_size=out_channels, num_layers=3,  # cat ablation
        self.mlp = MLP(input_size=latent_size, output_size=out_channels, num_layers=3,  # att and sum ablation
                       halving_size=False, dropout=0.3)

        self.lcp_preprocess = True

        self.activation = torch.nn.ReLU()

        print(f'Network -- backbone -- {count_parameters(self.encoder)} parameters')
        print(f'Network -- projection -- {count_parameters(self.projection)} parameters')
        print(f'Network -- point_net -- {count_parameters(self.point_net)} parameters')
        print(f'Network -- mlp -- {count_parameters(self.mlp)} parameters')

    def forward(self, data):
        latents = self.encoder.forward(data, spectral_only=True)
        data['latents'] = latents
        ret_data = self.from_latent(data)
        return ret_data

    def get_latent(self, data):
        latents = self.encoder.forward(data, spectral_only=False)
        data['latents'] = latents
        data['proj_correction'] = None
        return data

    def from_latent(self, data: typing.Dict[str, torch.Tensor]):
        feat_proj = self.projection.forward(data, has_proj_ids=False)

        # zero tensor for debug
        # feat_pn_shape = (data['proj_ids'].shape[0], data['proj_ids'].shape[2], data['proj_ids'].shape[1])
        # feat_pointnet = torch.zeros(feat_pn_shape, dtype=torch.float32, device=self.device)

        # PointNetFeat uses query points for batch dim -> need to flatten shape * query points dim
        pts_local_shape = data['pts_local_ps'].shape
        pts_local_flat_shape = (pts_local_shape[0] * pts_local_shape[1], pts_local_shape[2], pts_local_shape[3])
        pts_local_ps_flat = data['pts_local_ps'].view(pts_local_flat_shape)
        feat_pointnet_flat = self.point_net.forward(pts_local_ps_flat.transpose(1, 2), pts_weights=None)[0]
        feat_pointnet = feat_pointnet_flat.view((pts_local_shape[0], pts_local_shape[1], feat_pointnet_flat.shape[1]))

        # cat ablation
        # feat_all = torch.cat((feat_proj.transpose(1, 2), feat_pointnet), dim=2)

        # sum ablation -> vanilla
        feat_all = torch.sum(torch.stack((feat_proj.transpose(1, 2), feat_pointnet), dim=0), dim=0)
        # feat_all = feat_proj.transpose(1, 2) + feat_pointnet  # result is non-contiguous

        # # att: [batch, feat_len, num_feat] -> [batch, feat_len]
        # feat_all = torch.stack((feat_proj.transpose(1, 2), feat_pointnet), dim=3)
        # feat_all_shape = feat_all.shape
        # feat_all = feat_all.view(feat_all_shape[0] * feat_all_shape[1], feat_all_shape[2], feat_all_shape[3])
        # feat_all = self.branch_att.forward(feat_all)
        # feat_all = self.activation(feat_all)
        # feat_all = feat_all.view(feat_all_shape[0], feat_all_shape[1], feat_all_shape[2])

        # PointNetFeat uses query points for batch dim -> need to flatten shape * query points dim
        feat_all_flat = feat_all.view((feat_all.shape[0] * feat_all.shape[1], feat_all.shape[2]))
        ret_data_flat = self.mlp(feat_all_flat)
        ret_data = ret_data_flat.view((feat_all.shape[0], feat_all.shape[1], ret_data_flat.shape[1]))
        ret_data = ret_data.transpose(1, 2)

        return ret_data

