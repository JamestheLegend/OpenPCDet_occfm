# tools/create_carla_gtdb.py
import yaml
from pathlib import Path
from easydict import EasyDict
from pcdet.utils import common_utils
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset

def main():
    cfg_path = Path("tools/cfgs/dataset_configs/nuscenes_dataset.yaml")
    cfg = EasyDict(yaml.safe_load(open(cfg_path, "r")))

    cfg.VERSION = "v1.0-trainval"
    cfg.MAX_SWEEPS = 1

    # Use CARLA infos as "train"
    cfg.INFO_PATH["train"] = ["nuscenes_infos_1sweeps_train_carla.pkl"]

    dataset = NuScenesDataset(
        dataset_cfg=cfg,
        class_names=None,
        training=True,
        root_path=Path("data/nuscenes"),
        logger=common_utils.create_logger()
    )
    dataset.create_groundtruth_database(max_sweeps=1)

if __name__ == "__main__":
    main()
