# tools/create_carla_train_infos.py
from pathlib import Path
import pickle
from nuscenes.nuscenes import NuScenes
from pcdet.datasets.nuscenes import nuscenes_utils

def build_infos_for_one_record(record_dir: Path, max_sweeps: int):
    nusc = NuScenes(version="v1.0-nusc_like", dataroot=str(record_dir), verbose=False)

    # Put ALL scenes into train set (no val split for CARLA)
    train_scene_tokens = set([s["token"] for s in nusc.scene])
    val_scene_tokens = set()

    train_infos, _ = nuscenes_utils.fill_trainval_infos(
        data_path=record_dir,
        nusc=nusc,
        train_scenes=train_scene_tokens,
        val_scenes=val_scene_tokens,
        test=False,
        max_sweeps=max_sweeps,
        with_cam=False
    )
    return train_infos

def main():
    # Global nuScenes root used by OpenPCDet
    global_root = Path("data/nuscenes/v1.0-trainval").resolve()

    # Where your CARLA records live under that root
    carla_root = global_root / "carla" / "nusc_like_multi"
    assert carla_root.exists(), f"Missing {carla_root}"

    record_dirs = sorted([p for p in carla_root.iterdir() if p.is_dir() and p.name.startswith("record_")])
    assert len(record_dirs) > 0, "No record_* folders found"

    max_sweeps = 1

    all_infos = []
    for rec in record_dirs:
        infos = build_infos_for_one_record(rec, max_sweeps=max_sweeps)

        # IMPORTANT: rewrite lidar_path so it is relative to the GLOBAL root, not the record root
        # Example original: "samples/LIDAR_TOP/xxx.bin"
        # New: "carla/nusc_like_multi/record_xxx/samples/LIDAR_TOP/xxx.bin"
        prefix = Path("carla") / "nusc_like_multi" / rec.name
        for info in infos:
            info["lidar_path"] = str(prefix / info["lidar_path"])
            # sweeps also exist in info['sweeps'] (but for max_sweeps=1, it will be empty)
        all_infos.extend(infos)

        print(f"[OK] {rec.name}: {len(infos)} samples")

    out_path = global_root / f"nuscenes_infos_1sweeps_train_carla.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_infos, f)

    print("Saved:", out_path)
    print("Total CARLA train samples:", len(all_infos))

if __name__ == "__main__":
    main()
