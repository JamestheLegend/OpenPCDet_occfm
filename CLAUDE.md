# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenPCDet is a PyTorch-based toolbox for LiDAR-based 3D object detection from point clouds. It implements state-of-the-art detection methods including PointPillar, SECOND, PointRCNN, Part-A2, PV-RCNN, PV-RCNN++, VoxelRCNN, CenterPoint, VoxelNeXt, DSVT, TransFusion, BEVFusion, and MPPNet.

## Build and Installation

```bash
# Install the pcdet library and compile CUDA extensions
python setup.py develop
```

This compiles custom CUDA operations (iou3d_nms, roiaware_pool3d, roipoint_pool3d, pointnet2) required for the models.

**Dependencies:** spconv (sparse convolution library) must be installed separately. Use spconv v2.x with pip for PyTorch 1.3+.

## Training and Testing Commands

```bash
# Single GPU training
cd tools
python train.py --cfg_file cfgs/kitti_models/pointpillar.yaml

# Multi-GPU training (from tools/ directory)
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/kitti_models/pointpillar.yaml

# Testing with checkpoint
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CHECKPOINT}

# Multi-GPU testing
sh scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

## Dataset Preparation

Generate dataset info files before training:

```bash
# KITTI
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

# NuScenes
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval

# Waymo
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

## Architecture

### Module Topology

Models are built from a modular pipeline defined in `Detector3DTemplate.module_topology`:

```
vfe -> backbone_3d -> map_to_bev_module -> pfe -> backbone_2d -> dense_head -> point_head -> roi_head
```

Each module is optional and configured via YAML. The `build_networks()` method constructs the pipeline dynamically.

### Key Directories

- **pcdet/models/detectors/**: Complete detector implementations (e.g., `pointpillar.py`, `pv_rcnn.py`)
- **pcdet/models/backbones_3d/**: 3D feature extractors (spconv-based, PointNet2)
- **pcdet/models/backbones_2d/**: BEV feature extractors
- **pcdet/models/dense_heads/**: Detection heads (anchor-based, center-based)
- **pcdet/models/roi_heads/**: Second-stage refinement heads
- **pcdet/datasets/**: Dataset loaders (KITTI, NuScenes, Waymo, etc.)
- **pcdet/ops/**: Custom CUDA operations (NMS, pooling, PointNet2 ops)
- **tools/cfgs/**: Configuration files organized by dataset

### Configuration System

Configs use YAML with inheritance via `_BASE_CONFIG_`. Model configs in `tools/cfgs/{dataset}_models/` inherit from dataset configs in `tools/cfgs/dataset_configs/`.

Key config sections:
- `CLASS_NAMES`: Detection classes
- `DATA_CONFIG`: Dataset paths, point cloud range, voxelization, augmentation
- `MODEL`: Architecture specification (VFE, BACKBONE_3D, BACKBONE_2D, DENSE_HEAD, ROI_HEAD)
- `OPTIMIZATION`: Training hyperparameters

### Data Flow

1. Raw point cloud -> VFE (Voxel Feature Encoder) -> voxel features
2. Voxel features -> 3D Backbone (sparse conv) -> 3D feature volume
3. 3D features -> Map to BEV -> 2D bird's eye view features
4. BEV features -> 2D Backbone -> multi-scale features
5. Features -> Dense Head -> proposals/predictions
6. (Two-stage) Proposals -> ROI Head -> refined predictions

### 3D Box Format

Unified box representation: `(x, y, z, dx, dy, dz, heading)` - center coordinates, dimensions, and yaw angle.

## Supported Datasets

KITTI, NuScenes, Waymo, Lyft, Pandaset, ONCE, Argoverse2, and custom datasets via the template in `docs/CUSTOM_DATASET_TUTORIAL.md`.
