import argparse
import copy
from pathlib import Path

from quinine import Quinfig

from schema import schema
from train import main
from utils.multigpu_utils import init_distributed_mode


def list_of_strings(arg):
    return arg.split(",")


def mass_parser():

    parser = argparse.ArgumentParser(
        description="a light parser that leverages quinine parse"
    )
    parser.add_argument(
        "--quinine_config_path",
        type=str,
    )
    parser.add_argument("--local_rank", default=0, type=int)

    return parser


if __name__ == "__main__":
    raw_args = mass_parser().parse_args()
    template_args = Quinfig(
        config_path=raw_args.quinine_config_path, schema=schema
    )
    init_distributed_mode(template_args.multigpu)
    experiment_root = Path(__file__).absolute().parent
    yaml_list = [
        str(experiment_root / "4env_4d.yaml"),
        str(experiment_root / "100env_4d.yaml"),
        str(experiment_root / "inf_4d.yaml"),
        str(experiment_root / "16env_2d_sq.yaml"),
        str(experiment_root / "inf_2d_sq.yaml"),
    ]
    for yaml_path in yaml_list:
        print(yaml_path)
        temp = Quinfig(config_path=yaml_path, schema=schema)
        template_args_ = copy.deepcopy(template_args)
        temp.multigpu = template_args_.multigpu
        main(temp)
