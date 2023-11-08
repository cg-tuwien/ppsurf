import typing

from pytorch_lightning import cli

from source.poco_model import PocoModel
from source.occupancy_data_module import OccupancyDataModule

from source.cli import CLI


class PocoCLI(CLI):

    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)

        parser.link_arguments('data.init_args.in_file', 'model.init_args.in_file')
        parser.link_arguments('data.init_args.padding_factor', 'model.init_args.padding_factor')

        # this direction because logger is not available for test/predict
        parser.link_arguments('model.init_args.name', 'trainer.logger.init_args.name')

    def handle_rec_subcommand(self, args: typing.List[str]) -> typing.List[str]:
        """Replace 'rec' subcommand with predict and its default parameters.
        Download model if necessary.
        """
        raise NotImplementedError()


def cli_main():
    PocoCLI(model_class=PocoModel, subclass_mode_model=True,
            datamodule_class=OccupancyDataModule, subclass_mode_data=True)


if __name__ == '__main__':
    # for testing
    # sys.argv = ['poco.py', 'fit',
    #             '-c', 'configs/poco.yaml',
    #             # '--print_config'
    #             ]

    # Run PPS, run!
    cli_main()
