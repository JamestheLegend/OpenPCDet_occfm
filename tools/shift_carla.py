#!/usr/bin/env python3
import os
import json
import glob
import argparse
import numpy as np

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def norm_rel_path(s: str) -> str:
    # sample_data["filename"] may contain backslashes from Windows
    return s.replace("\\", "/").lstrip("/")

def shift_bin_inplace(bin_path: str, point_dim: int, dz: float) -> int:
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size % point_dim != 0:
        raise ValueError(f"[BAD BIN SHAPE] {bin_path}: size={arr.size} not divisible by point_dim={point_dim}")
    pts = arr.reshape(-1, point_dim)
    pts[:, 2] += dz
    pts.astype(np.float32).tofile(bin_path)
    return pts.shape[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla_root", type=str, required=True,
                    help="Path to data/carla/nusc_like_multi")
    ap.add_argument("--delta_z", type=float, default=0.4)
    ap.add_argument("--point_dim", type=int, default=5, choices=[4, 5])
    ap.add_argument("--version_dir", type=str, default="v1.0-nusc_like",
                    help="nuScenes tables folder name inside each record")
    ap.add_argument("--lidar_channel", type=str, default="LIDAR_TOP")
    ap.add_argument("--force", action="store_true",
                    help="Allow shifting again even if marker exists (DANGEROUS)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print what would change, do not write files")
    args = ap.parse_args()

    carla_root = os.path.abspath(args.carla_root)
    assert os.path.isdir(carla_root), f"Not a folder: {carla_root}"

    record_dirs = sorted([p for p in glob.glob(os.path.join(carla_root, "record_*")) if os.path.isdir(p)])
    if not record_dirs:
        raise RuntimeError(f"No record_* folders found under {carla_root}")

    print(f"[INFO] Found {len(record_dirs)} records under: {carla_root}")
    print(f"[INFO] delta_z={args.delta_z}, point_dim={args.point_dim}, version_dir={args.version_dir}")

    total_bins = 0
    total_points = 0
    total_ann = 0

    for rec in record_dirs:
        marker = os.path.join(rec, f".zshifted_{args.delta_z:+.3f}")
        if os.path.exists(marker) and not args.force:
            print(f"[SKIP] {os.path.basename(rec)} already shifted (marker exists): {marker}")
            continue

        ver = os.path.join(rec, args.version_dir)
        if not os.path.isdir(ver):
            print(f"[WARN] {os.path.basename(rec)} missing version dir: {ver} (skipping)")
            continue

        sample_data_p = os.path.join(ver, "sample_data.json")
        sample_ann_p  = os.path.join(ver, "sample_annotation.json")

        if not os.path.isfile(sample_data_p):
            print(f"[WARN] Missing: {sample_data_p} (skipping record)")
            continue
        if not os.path.isfile(sample_ann_p):
            print(f"[WARN] Missing: {sample_ann_p} (skipping record)")
            continue

        sample_data = load_json(sample_data_p)
        sample_ann  = load_json(sample_ann_p)

        # --- Shift LiDAR bins ---
        def is_lidar_row(sd):
            fn = str(sd.get("filename", "")).lower()
            # robust: your sample_data.json has channel/modality as None
            return (
                fn.startswith("samples/lidar_top/") or
                fn.startswith("sweeps/lidar_top/") or
                ("/lidar_" in fn and fn.endswith(".bin")) or
                ("lidar" in fn and fn.endswith(".bin"))
            )

        lidar_entries = [x for x in sample_data if is_lidar_row(x)]

        if not lidar_entries:
            print(f"[WARN] {os.path.basename(rec)}: no channel={args.lidar_channel} entries found in sample_data.json")
        else:
            print(f"[REC] {os.path.basename(rec)}: shifting {len(lidar_entries)} lidar frames")

        rec_bins = 0
        rec_points = 0
        for sd in lidar_entries:
            fn = sd.get("filename", "")
            if not fn:
                continue
            fn = norm_rel_path(fn)

            # Most exporters store paths relative to record root, e.g. "sweeps/LIDAR_TOP/xxx.bin"
            # If yours already includes "record_xxx/...", norm_rel_path keeps it but we handle both:
            candidate1 = os.path.join(rec, fn)
            candidate2 = os.path.join(carla_root, fn)  # fallback if filename is rooted at carla_root

            if os.path.isfile(candidate1):
                bin_path = candidate1
            elif os.path.isfile(candidate2):
                bin_path = candidate2
            else:
                # last resort: try to locate by basename inside record
                base = os.path.basename(fn)
                hits = glob.glob(os.path.join(rec, "**", base), recursive=True)
                if len(hits) == 1 and os.path.isfile(hits[0]):
                    bin_path = hits[0]
                else:
                    raise FileNotFoundError(
                        f"Could not locate lidar bin for filename='{sd.get('filename')}'\n"
                        f"Tried:\n  {candidate1}\n  {candidate2}\n"
                        f"and basename search hits={len(hits)}"
                    )

            if args.dry_run:
                rec_bins += 1
                continue

            npts = shift_bin_inplace(bin_path, args.point_dim, args.delta_z)
            rec_bins += 1
            rec_points += npts

        # --- Shift annotations (global z) ---
        # This will make lidar-frame boxes also shift ~+dz after global->lidar transform (assuming no crazy roll/pitch).
        ann_count = 0
        for ann in sample_ann:
            tr = ann.get("translation", None)
            if isinstance(tr, list) and len(tr) == 3:
                if not args.dry_run:
                    tr[2] = float(tr[2]) + float(args.delta_z)
                    ann["translation"] = tr
                ann_count += 1

        if not args.dry_run:
            save_json(sample_ann_p, sample_ann)
            # marker
            with open(marker, "w", encoding="utf-8") as f:
                f.write("shifted\n")

        total_bins += rec_bins
        total_points += rec_points
        total_ann += ann_count

        if args.dry_run:
            print(f"  [DRY] would shift bins={rec_bins}, would shift annotations={ann_count}")
        else:
            print(f"  [OK] shifted bins={rec_bins}, points={rec_points}, annotations={ann_count}")

    print("\n[SUMMARY]")
    if args.dry_run:
        print(f"Would shift total lidar bins: {total_bins}")
        print(f"Would shift total annotations: {total_ann}")
    else:
        print(f"Shifted total lidar bins: {total_bins}")
        print(f"Shifted total points: {total_points}")
        print(f"Shifted total annotations: {total_ann}")
    print("[DONE]")

if __name__ == "__main__":
    main()
