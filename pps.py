import sys
import os
import typing

from pytorch_lightning import cli

from source.poco_model import PocoModel
from source.occupancy_data_module import OccupancyDataModule

from poco import PocoCLI

# run with:
# python pps.py fit
# python pps.py validate
# python pps.py test
# python pps.py predict
# configs as below


class PPSCLI(PocoCLI):

    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)

        parser.link_arguments('model.init_args.num_pts_local', 'data.init_args.num_pts_local')

    def handle_rec_subcommand(self, args: typing.List[str]) -> typing.List[str]:
        """Replace 'rec' subcommand with predict and its default parameters.
        Download model if necessary.
        """

        # no rec -> nothing to do
        if len(args) <= 1 or args[1] != 'rec':
            return args

        # check syntax
        if len(args) < 4 or args[0] != os.path.basename(__file__):
            raise ValueError(
                'Invalid syntax for rec subcommand: {}\n'
                'Make sure that it matches this example: '
                'pps.py rec in_file.ply out_file.ply --model.init_args.rec_batch_size 50000'.format(' '.join(sys.argv)))

        in_file = args[2]
        if not os.path.exists(in_file):
            raise ValueError('Input file does not exist: {}'.format(in_file))
        out_dir = args[3]
        os.makedirs(out_dir, exist_ok=True)
        extra_params = args[4:]
        model_path = os.path.join('models/ppsurf_50nn/version_0/checkpoints/last.ckpt')

        # assemble predict subcommand
        args_pred = args[:1]
        args_pred += [
            'predict',
            '-c', 'configs/poco.yaml',
            '-c', 'configs/ppsurf.yaml',
            '-c', 'configs/ppsurf_50nn.yaml',
            '--ckpt_path', model_path,
            '--data.init_args.in_file', in_file,
            '--model.init_args.results_dir', out_dir,
            '--trainer.logger', 'False',
            '--trainer.devices', '1'
        ]
        args_pred += extra_params
        print('Converted rec subcommand to predict subcommand: {}'.format(' '.join(args_pred)))

        # download model if necessary
        if not os.path.exists(model_path):
            print('Model checkpoint not found at {}. Downloading...'.format(model_path))
            os.system('python models/download_ppsurf_50nn.py')

        return args_pred


def cli_main():
    PPSCLI(model_class=PocoModel, subclass_mode_model=True,
           datamodule_class=OccupancyDataModule, subclass_mode_data=True)


def fixed_cmd():
    # for debugging

    # train
    sys.argv = ['pps.py',
                'fit',
                '-c', 'configs/poco.yaml',
                '-c', 'configs/ppsurf.yaml',
                '-c', 'configs/ppsurf_mini.yaml',
                # '--debug', 'True',
                # '--print_config'
                ]
    cli_main()

    # test
    sys.argv = ['pps.py',
                'test',
                '-c', 'configs/poco.yaml',
                '-c', 'configs/ppsurf.yaml',
                '-c', 'configs/ppsurf_mini.yaml',
                '--ckpt_path', 'models/ppsurf_mini/version_0/checkpoints/last.ckpt', '--trainer.logger', 'False',
                # '--print_config'
                ]
    cli_main()

    # predict
    sys.argv = ['pps.py',
                'predict',
                '-c', 'configs/poco.yaml',
                '-c', 'configs/ppsurf.yaml',
                '-c', 'configs/ppsurf_mini.yaml',
                '--ckpt_path', 'models/ppsurf_mini/version_0/checkpoints/last.ckpt', '--trainer.logger', 'False',
                # '--print_config'
                ]
    cli_main()

    # rec
    sys.argv = ['pps.py',
                'rec',
                'datasets/abc_minimal/04_pts_vis/00011084_fddd53ce45f640f3ab922328_trimesh_019.xyz.ply',
                'results/rec/test/00011084_fddd53ce45f640f3ab922328_trimesh_019.ply',
                ]
    cli_main()


if __name__ == '__main__':
    # fixed_cmd()
    cli_main()
