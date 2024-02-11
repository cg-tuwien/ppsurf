import typing

import pytorch_lightning as pl
import torch
from torch.nn import functional as f


# https://github.com/numpy/numpy/issues/5228
def cartesian_to_polar(pts_cart: torch.Tensor):
    batch_size = pts_cart.shape[0]
    num_pts = pts_cart.shape[1]
    num_dim = pts_cart.shape[2]
    pts_cart_flat = pts_cart.reshape((-1, num_dim))

    def pol_2d():
        x = pts_cart_flat[:, 0]
        y = pts_cart_flat[:, 1]

        r = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        return torch.stack((r, phi), dim=1)

    def pol_3d():
        x = pts_cart_flat[:, 0]
        y = pts_cart_flat[:, 1]
        z = pts_cart_flat[:, 2]

        hxy = torch.hypot(x, y)
        r = torch.hypot(hxy, z)
        el = torch.atan2(z, hxy)
        az = torch.atan2(y, x)
        return torch.stack((az, el, r), dim=1)

    pts_spherical_flat = pol_2d() if num_dim == 2 else pol_3d()
    pts_spherical = pts_spherical_flat.reshape((batch_size, num_pts, num_dim))

    return pts_spherical


def pos_encoding(pts: torch.Tensor, pos_encoding_levels: int, skip_last_dim=False):
    """
    use positional encoding on points
    3d example: [x, y, z] -> [f(cos, x), f(cos, y), f(cos, z), f(sin, x), f(sin, y), f(sin, z)]
    :param pts: tensor[b, n, 2 or 3]
    :param pos_encoding_levels: int
    :param skip_last_dim: bool, skip last dim of input points (necessary for radius of polar coordinates)
    :return:
    """

    if pos_encoding_levels <= 0:
        return pts

    batch_size = pts.shape[0]
    num_pts = pts.shape[1]
    num_dim = pts.shape[2]
    num_dim_out = num_dim * 2 * pos_encoding_levels
    pts_enc = torch.zeros((batch_size, num_pts, num_dim_out), device=pts.device)

    for dim in range(num_dim):
        for lvl in range(pos_encoding_levels):
            dim_out = dim * lvl * 2
            if skip_last_dim and dim == num_dim - 1:
                pts_enc[..., dim_out] = pts[..., dim]
                pts_enc[..., dim_out + num_dim] = pts[..., dim]
            else:
                pts_enc[..., dim_out] = torch.cos(pts[..., dim] * lvl * torch.pi * pow(2.0, lvl))
                pts_enc[..., dim_out + num_dim] = torch.sin(pts[..., dim] * lvl * torch.pi * pow(2.0, lvl))

    return pts_enc


class AttentionPoco(pl.LightningModule):
    # self-attention for feature vectors
    # adapted from POCO attention
    # https://github.com/valeoai/POCO/blob/4e39b5e722c82e91570df5f688e2c6e4870ffe65/networks/decoder/interp_attention.py

    def __init__(self, net_size_max=1024, reduce=True):
        super(AttentionPoco, self).__init__()

        self.fc_query = torch.nn.Conv2d(net_size_max, 1, 1)
        self.fc_value = torch.nn.Conv2d(net_size_max, net_size_max, 1)
        self.reduce = reduce

    def forward(self, feature_vectors: torch.Tensor):
        # [feat_len, batch, num_feat] expected -> feature dim to dim 0
        feature_vectors_t = torch.permute(feature_vectors, (1, 0, 2))

        query = self.fc_query(feature_vectors_t).squeeze(0)  # fc over feature dim -> [batch, num_feat]
        value = self.fc_value(feature_vectors_t).permute(1, 2, 0)  # -> [batch, num_feat, feat_len]

        weights = torch.nn.functional.softmax(query, dim=-1)  # softmax over num_feat -> [batch, num_feat]
        if self.reduce:
            feature_vector_out = torch.sum(value * weights.unsqueeze(-1).broadcast_to(value.shape), dim=1)
        else:
            feature_vector_out = (weights.unsqueeze(2) * value).permute(0, 2, 1)
        return feature_vector_out


def batch_quat_to_rotmat(q, out=None):
    """
    quaternion a + bi + cj + dk should be given in the form [a,b,c,d]
    :param q:
    :param out:
    :return:
    """

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2 / torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out


class STN(pl.LightningModule):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.net_size_max = net_size_max
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.net_size_max, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = torch.nn.Linear(self.net_size_max, int(self.net_size_max / 2))
        self.fc2 = torch.nn.Linear(int(self.net_size_max / 2), int(self.net_size_max / 4))
        self.fc3 = torch.nn.Linear(int(self.net_size_max / 4), self.dim*self.dim)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.net_size_max)
        self.bn4 = torch.nn.BatchNorm1d(int(self.net_size_max / 2))
        self.bn5 = torch.nn.BatchNorm1d(int(self.net_size_max / 4))

        if self.num_scales > 1:
            self.fc0 = torch.nn.Linear(self.net_size_max * self.num_scales, self.net_size_max)
            self.bn0 = torch.nn.BatchNorm1d(self.net_size_max)

    def forward(self, x):
        batch_size = x.size()[0]
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), self.net_size_max * self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*self.net_size_max:(s+1)*self.net_size_max, :] = \
                    self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, self.net_size_max*self.num_scales)

        if self.num_scales > 1:
            x = f.relu(self.bn0(self.fc0(x)))

        x = f.relu(self.bn4(self.fc1(x)))
        x = f.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x


class QSTN(pl.LightningModule):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.net_size_max = net_size_max
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.net_size_max, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = torch.nn.Linear(self.net_size_max, int(self.net_size_max / 2))
        self.fc2 = torch.nn.Linear(int(self.net_size_max / 2), int(self.net_size_max / 4))
        self.fc3 = torch.nn.Linear(int(self.net_size_max / 4), 4)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.net_size_max)
        self.bn4 = torch.nn.BatchNorm1d(int(self.net_size_max / 2))
        self.bn5 = torch.nn.BatchNorm1d(int(self.net_size_max / 4))

        if self.num_scales > 1:
            self.fc0 = torch.nn.Linear(self.net_size_max*self.num_scales, self.net_size_max)
            self.bn0 = torch.nn.BatchNorm1d(self.net_size_max)

    def forward(self, x):
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), self.net_size_max*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*self.net_size_max:(s+1)*self.net_size_max, :] = \
                    self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, self.net_size_max*self.num_scales)

        if self.num_scales > 1:
            x = f.relu(self.bn0(self.fc0(x)))

        x = f.relu(self.bn4(self.fc1(x)))
        x = f.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x_quat = x + iden

        # convert quaternion to rotation matrix
        x = batch_quat_to_rotmat(x_quat)

        return x, x_quat


class PointNetfeat(pl.LightningModule):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500,
                 polar=False, use_point_stn=True, use_feat_stn=True,
                 output_size=100, sym_op='max', dim=3):
        super(PointNetfeat, self).__init__()

        self.net_size_max = net_size_max
        self.num_points = num_points
        self.num_scales = num_scales
        self.polar = polar
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.output_size = output_size
        self.dim = dim

        if self.use_point_stn:
            self.stn1 = QSTN(net_size_max=net_size_max, num_scales=self.num_scales,
                             num_points=num_points, dim=dim, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(net_size_max=net_size_max, num_scales=self.num_scales,
                            num_points=num_points, dim=64, sym_op=self.sym_op)

        self.conv0a = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)
        self.bn0a = torch.nn.BatchNorm1d(64)
        self.bn0b = torch.nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, output_size, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(output_size)

        if self.num_scales > 1:
            self.conv4 = torch.nn.Conv1d(output_size, output_size*self.num_scales, 1)
            self.bn4 = torch.nn.BatchNorm1d(output_size*self.num_scales)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            pass
        elif self.sym_op == 'wsum':
            pass
        elif self.sym_op == 'att':
            self.att = AttentionPoco(output_size)
        else:
            raise ValueError('Unsupported symmetric operation: {}'.format(self.sym_op))

    def forward(self, x, pts_weights):

        # input transform
        if self.use_point_stn:
            trans, trans_quat = self.stn1(x[:, :3, :])  # transform only point data
            # an error here can mean that your input size is wrong (e.g. added normals in the point cloud files)
            x_transformed = torch.bmm(trans, x[:, :3, :])  # transform only point data
            x = torch.cat((x_transformed, x[:, 3:, :]), dim=1)
        else:
            trans = None
            trans_quat = None

        if bool(self.polar):
            x = torch.permute(x, (0, 2, 1))
            x = cartesian_to_polar(pts_cart=x)
            x = torch.permute(x, (0, 2, 1))

        # mlp (64,64)
        x = f.relu(self.bn0a(self.conv0a(x)))
        x = f.relu(self.bn0b(self.conv0b(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = torch.bmm(trans2, x)
        else:
            trans2 = None

        # mlp (64,128,output_size)
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # mlp (output_size,output_size*num_scales)
        if self.num_scales > 1:
            x = self.bn4(self.conv4(f.relu(x)))

        # symmetric max operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            elif self.sym_op == 'wsum':
                pts_weights_bc = torch.broadcast_to(torch.unsqueeze(pts_weights, 1), size=x.shape)
                x = x * pts_weights_bc
                x = torch.sum(x, 2, keepdim=True)
            elif self.sym_op == 'att':
                x = self.att(x)
            else:
                raise ValueError('Unsupported symmetric operation: {}'.format(self.sym_op))

        else:
            x_scales = x.new_empty(x.size(0), self.output_size*self.num_scales**2, 1)
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*self.output_size:(s+1)*self.num_scales*self.output_size, :] = \
                        self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*self.output_size:(s+1)*self.num_scales*self.output_size, :] = \
                        torch.sum(x[:, :, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % self.sym_op)
            x = x_scales

        x = x.view(-1, self.output_size * self.num_scales ** 2)

        return x, trans, trans_quat, trans2


class MLP(pl.LightningModule):
    def __init__(self, input_size: int, output_size: int, num_layers: int,
                 halving_size=True, final_bn_act=False, final_layer_norm=False,
                 activation: typing.Optional[typing.Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 norm: typing.Optional[typing.Callable[..., torch.nn.Module]] = torch.nn.BatchNorm1d,
                 fc_layer=torch.nn.Linear, dropout=0.0):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        if halving_size:
            layer_sizes = [int(input_size / (2 ** i)) for i in range(num_layers)]
        else:
            layer_sizes = [input_size for _ in range(num_layers)]

        fully_connected = [fc_layer(layer_sizes[i], layer_sizes[i+1]) for i in range(num_layers-1)]
        norms = [norm(layer_sizes[i + 1]) for i in range(num_layers - 1)]

        layers_list = []
        for i in range(self.num_layers-1):
            layers_list.append(torch.nn.Sequential(
                fully_connected[i],
                norms[i],
                activation(),
                torch.nn.Dropout(dropout),
            ))

        final_modules = [fc_layer(layer_sizes[-1], output_size)]
        if final_bn_act:
            if final_layer_norm:
                final_modules.append(torch.nn.LayerNorm(output_size))
            else:
                final_modules.append(norm(output_size))
            final_modules.append(activation())
        final_layer = torch.nn.Sequential(*final_modules)
        layers_list.append(final_layer)

        self.layers = torch.nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.layers.forward(x)
        return x


class ResidualBlock(pl.LightningModule):

    def __init__(self, in_channels, out_channels, kernel_size, activation=torch.nn.ReLU()):
        super().__init__()
        bn = torch.nn.BatchNorm1d

        self.cv0 = torch.nn.Conv1d(in_channels, in_channels // 2, 1)
        self.bn0 = bn(in_channels // 2)
        self.cv1 = FKAConvLayer(in_channels // 2, in_channels // 2, kernel_size, activation=activation)
        self.bn1 = bn(in_channels // 2)
        self.cv2 = torch.nn.Conv1d(in_channels // 2, out_channels, 1)
        self.bn2 = bn(out_channels)
        self.activation = torch.nn.ReLU(inplace=True)

        self.shortcut = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels \
            else torch.nn.Identity()
        self.bn_shortcut = bn(out_channels) if in_channels != out_channels else torch.nn.Identity()

    def forward(self, x, pts, support_points, neighbors_indices):
        x_short = x
        x = self.activation(self.bn0(self.cv0(x)))
        x = self.activation(self.bn1(self.cv1(x, pts, support_points, neighbors_indices)))
        x = self.bn2(self.cv2(x))

        x_short = self.bn_shortcut(self.shortcut(x_short))
        if x_short.shape[2] != x.shape[2]:
            x_short = max_pool(x_short, neighbors_indices)

        x = self.activation(x + x_short)

        return x


class FKAConvNetwork(pl.LightningModule):

    def __init__(self, in_channels, out_channels, segmentation=False, hidden=64, dropout=0.5,
                 last_layer_additional_size=None, fix_support_number=False,
                 activation=torch.nn.ReLU(), x4d_bug_fixed=False):
        super().__init__()

        self.fixed = x4d_bug_fixed

        self.lcp_preprocess = True
        self.segmentation = segmentation
        self.fix_support_point_number = fix_support_number
        self.kernel_size = 16

        self.cv0 = FKAConvLayer(in_channels, hidden, 16, activation=activation)

        bn = torch.nn.BatchNorm1d
        self.bn0 = bn(hidden)

        def _make_resnet_block(in_channels_resnetb, out_channels_resnetb):
            return ResidualBlock(in_channels=in_channels_resnetb, out_channels=out_channels_resnetb,
                                 kernel_size=self.kernel_size, activation=activation)

        self.resnetb01 = _make_resnet_block(hidden, hidden)
        self.resnetb10 = _make_resnet_block(hidden, 2 * hidden)
        self.resnetb11 = _make_resnet_block(2 * hidden, 2 * hidden)
        self.resnetb20 = _make_resnet_block(2 * hidden, 4 * hidden)
        self.resnetb21 = _make_resnet_block(4 * hidden, 4 * hidden)
        self.resnetb30 = _make_resnet_block(4 * hidden, 8 * hidden)
        self.resnetb31 = _make_resnet_block(8 * hidden, 8 * hidden)
        self.resnetb40 = _make_resnet_block(8 * hidden, 16 * hidden)
        self.resnetb41 = _make_resnet_block(16 * hidden, 16 * hidden)
        if self.segmentation:

            self.cv5 = torch.nn.Conv1d(32 * hidden, 16 * hidden, 1)
            self.bn5 = bn(16 * hidden)
            self.cv3d = torch.nn.Conv1d(24 * hidden, 8 * hidden, 1)
            self.bn3d = bn(8 * hidden)
            self.cv2d = torch.nn.Conv1d(12 * hidden, 4 * hidden, 1)
            self.bn2d = bn(4 * hidden)
            self.cv1d = torch.nn.Conv1d(6 * hidden, 2 * hidden, 1)
            self.bn1d = bn(2 * hidden)
            self.cv0d = torch.nn.Conv1d(3 * hidden, hidden, 1)
            self.bn0d = bn(hidden)

            if last_layer_additional_size is not None:
                self.fcout = torch.nn.Conv1d(hidden + last_layer_additional_size, out_channels, 1)
            else:
                self.fcout = torch.nn.Conv1d(hidden, out_channels, 1)
        else:
            self.fcout = torch.nn.Conv1d(16 * hidden, out_channels, 1)

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, data, spectral_only=False):
        if not spectral_only:
            from source.poco_data_loader import get_fkaconv_ids
            spatial_data = get_fkaconv_ids(data)
            for key, value in spatial_data.items():
                data[key] = value

        # x = data['x']
        pts = data['pts']
        x = torch.ones_like(pts)

        x0 = self.activation(self.bn0(self.cv0(x, pts, pts, data['ids00'])))
        x0 = self.resnetb01(x0, pts, pts, data['ids00'])
        x1 = self.resnetb10(x0, pts, data['support1'], data['ids01'])
        x1 = self.resnetb11(x1, data['support1'], data['support1'], data['ids11'])
        x2 = self.resnetb20(x1, data['support1'], data['support2'], data['ids12'])
        x2 = self.resnetb21(x2, data['support2'], data['support2'], data['ids22'])
        x3 = self.resnetb30(x2, data['support2'], data['support3'], data['ids23'])
        x3 = self.resnetb31(x3, data['support3'], data['support3'], data['ids33'])
        x4 = self.resnetb40(x3, data['support3'], data['support4'], data['ids34'])
        x4 = self.resnetb41(x4, data['support4'], data['support4'], data['ids44'])

        if self.segmentation:
            x5 = x4.max(dim=2, keepdim=True)[0].expand_as(x4)
            x4d = self.activation(self.bn5(self.cv5(torch.cat([x4, x5], dim=1))))
            if not self.fixed:
                x4d = x4

            x3d = interpolate(x4d, data['ids43'])
            x3d = self.activation(self.bn3d(self.cv3d(torch.cat([x3d, x3], dim=1))))

            x2d = interpolate(x3d, data['ids32'])
            x2d = self.activation(self.bn2d(self.cv2d(torch.cat([x2d, x2], dim=1))))

            x1d = interpolate(x2d, data['ids21'])
            x1d = self.activation(self.bn1d(self.cv1d(torch.cat([x1d, x1], dim=1))))

            xout = interpolate(x1d, data['ids10'])
            xout = self.activation(self.bn0d(self.cv0d(torch.cat([xout, x0], dim=1))))
            xout = self.dropout(xout)
            xout = self.fcout(xout)
        else:
            xout = x4
            xout = self.dropout(xout)
            xout = self.fcout(xout)
            xout = xout.mean(dim=2)
        return xout


class FKAConvLayer(pl.LightningModule):

    def __init__(self, in_channels, out_channels, kernel_size=16, bias=False, dim=3,
                 activation=torch.nn.ReLU()):
        super().__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.dim = dim

        # convolution kernel
        self.cv = torch.nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=bias)

        # normalization radius
        self.norm_radius_momentum = 0.1
        self.register_buffer('norm_radius', torch.Tensor(1,))
        self.alpha = torch.nn.Parameter(torch.Tensor(1,), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.Tensor(1,), requires_grad=True)
        torch.nn.init.ones_(self.norm_radius.data)
        torch.nn.init.ones_(self.alpha.data)
        torch.nn.init.ones_(self.beta.data)

        # features to kernel weights
        self.fc1 = torch.nn.Conv2d(self.dim, self.kernel_size, 1, bias=False)
        self.fc2 = torch.nn.Conv2d(2 * self.kernel_size, self.kernel_size, 1, bias=False)
        self.fc3 = torch.nn.Conv2d(2 * self.kernel_size, self.kernel_size, 1, bias=False)
        self.bn1 = torch.nn.InstanceNorm2d(self.kernel_size, affine=True)
        self.bn2 = torch.nn.InstanceNorm2d(self.kernel_size, affine=True)

        self.activation = activation

    # TODO: try sigmoid again
    def forward(self, x, pts, support_points, neighbors_indices):

        if x is None:
            return None

        pts = batch_gather(pts, dim=2, index=neighbors_indices).contiguous()
        x = batch_gather(x, dim=2, index=neighbors_indices).contiguous()

        # center the neighborhoods (local coordinates)
        pts = pts - support_points.unsqueeze(3)

        # normalize points
        # compute distances from points to their support point
        distances = torch.sqrt((pts.detach() ** 2).sum(1))

        # update the normalization radius
        if self.training:
            mean_radius = distances.max(2)[0].mean()
            self.norm_radius.data = (
                    self.norm_radius.data * (1 - self.norm_radius_momentum)
                    + mean_radius * self.norm_radius_momentum
            )

        # normalize
        pts = pts / self.norm_radius

        # estimate distance weights
        distance_weight = torch.sigmoid(-self.alpha * distances + self.beta)
        distance_weight_s = distance_weight.sum(2, keepdim=True)
        distance_weight_s = distance_weight_s + (distance_weight_s == 0) + 1e-6
        distance_weight = (
            distance_weight / distance_weight_s * distances.shape[2]
        ).unsqueeze(1)

        # feature weighting matrix estimation
        if pts.shape[3] == 1:
            mat = self.activation(self.fc1(pts))
        else:
            mat = self.activation(self.bn1(self.fc1(pts)))
        mp1 = torch.max(mat * distance_weight, dim=3, keepdim=True)[0].expand(
            (-1, -1, -1, mat.shape[3])
        )
        mat = torch.cat([mat, mp1], dim=1)
        if pts.shape[3] == 1:
            mat = self.activation(self.fc2(mat))
        else:
            mat = self.activation(self.bn2(self.fc2(mat)))
        mp2 = torch.max(mat * distance_weight, dim=3, keepdim=True)[0].expand(
            (-1, -1, -1, mat.shape[3])
        )
        mat = torch.cat([mat, mp2], dim=1)
        mat = self.activation(self.fc3(mat)) * distance_weight
        # mat = torch.sigmoid(self.fc3(mat)) * distance_weight

        # compute features
        features = torch.matmul(
            x.transpose(1, 2), mat.permute(0, 2, 3, 1)
        ).transpose(1, 2)
        features = self.cv(features).squeeze(3)

        return features


@torch.jit.script
def batch_gather(data: torch.Tensor, dim: int, index: torch.Tensor):

    index_shape = list(index.shape)
    input_shape = list(data.shape)

    views = [data.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(data.shape))
    ]
    expanse = list(data.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)

    output = torch.gather(data, dim, index)

    # compute final shape
    output_shape = input_shape[0:dim] + index_shape[1:] + input_shape[dim+1:]

    return output.reshape(output_shape)


def max_pool(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    features = batch_gather(data, dim=2, index=indices).contiguous()
    features = features.max(dim=3)[0]
    return features


# TODO: test sum
def interpolate(x, neighbors_indices, method='mean'):

    mask = (neighbors_indices > -1)
    neighbors_indices[~mask] = 0

    x = batch_gather(x, 2, neighbors_indices)

    if neighbors_indices.shape[-1] > 1:
        if method == 'mean':
            return x.mean(-1)
        elif method == 'max':
            return x.mean(-1)[0]
    else:
        return x.squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
