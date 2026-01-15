# Implementation Plan: Train VoxelRCNN on CARLA Data, Evaluate on NuScenes

## Summary of Data Analysis

### CARLA Data Structure (`data/nusc_like_multi/`)
- **8 records** with ~3,516 samples each (total ~28,130 samples)
- **NuScenes-like JSON metadata** in `v1.0-nusc_like/` folder
- **Point cloud format**: 5 features (x, y, z, intensity, ring/timestamp) - range: x=[-96,96], y=[-61,73], z=[-7,12]
- **Categories**: Only `vehicle.car` and `human.pedestrian.adult`
- **Sensor setup**: LIDAR_TOP at [0, 0, 2.0] relative to ego with identity rotation
- **No sweeps**: sweeps folder is empty (single-frame data)
- **Annotations**: In global frame, need transformation to lidar frame

### Key Differences from NuScenes
1. No multi-sweep data (single frame only)
2. No camera data
3. Simpler category set (only car + pedestrian)
4. Boxes in global coordinates need transformation

---

## Implementation Steps

### Phase 1: Create Info File Generator Script

**File**: `tools/create_carla_infos.py`

This script will:
1. Load all 8 records' JSON metadata
2. Build lookup tables for samples, annotations, ego_poses, calibrated_sensors
3. Transform annotations from global frame → lidar frame using:
   - Ego pose (translation, rotation quaternion)
   - Calibrated sensor offset [0, 0, 2.0]
4. Generate OpenPCDet-compatible info pickle files:
   - `carla_infos_train.pkl` (all 8 records combined)
5. Generate ground truth database for data augmentation:
   - `carla_dbinfos_train.pkl`
   - `gt_database/` folder with cropped point clouds per object

**Box transformation logic**:
```
Global Frame → Ego Frame → Lidar Frame
- Subtract ego translation, apply inverse ego rotation
- Subtract sensor translation [0, 0, 2.0], apply inverse sensor rotation (identity)
- Extract yaw from resulting quaternion
- Convert size from [width, length, height] to [dx, dy, dz] = [length, width, height]
```

**Output info dict structure**:
```python
{
    'lidar_path': 'record_xxx/samples/LIDAR_TOP/xxx.bin',
    'token': 'sample_token',
    'sweeps': [],  # Empty, no sweeps
    'gt_boxes': np.array(N, 7),  # [x, y, z, dx, dy, dz, yaw] in lidar frame
    'gt_names': np.array(N,),    # ['car', 'car', ...]
    'num_lidar_pts': np.array(N,),
}
```

### Phase 2: Create Dataset Configuration

**File**: `tools/cfgs/dataset_configs/carla_dataset.yaml`

Key settings:
- `DATASET: 'NuScenesDataset'` (reuse existing loader with custom info files)
- `DATA_PATH: '../data/nusc_like_multi'`
- `VERSION: ''` (no version subfolder needed)
- `MAX_SWEEPS: 1` (single frame)
- `PRED_VELOCITY: False` (no velocity in CARLA data)
- `POINT_CLOUD_RANGE: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]` (match NuScenes for eval compatibility)
- `VOXEL_SIZE: [0.1, 0.1, 0.2]`

### Phase 3: Create VoxelRCNN Model Configuration

**File**: `tools/cfgs/carla_models/voxel_rcnn_car.yaml`

Key settings:
- `CLASS_NAMES: ['car']` (vehicle only)
- Inherit from carla_dataset.yaml
- VoxelRCNN architecture with:
  - MeanVFE
  - VoxelBackBone8x
  - HeightCompression (MAP_TO_BEV)
  - BaseBEVBackbone
  - AnchorHeadSingle (for first stage)
  - VoxelRCNNHead (for refinement)
- Anchor size: `[4.7, 2.1, 1.7]` (typical car dimensions)

### Phase 4: Create Evaluation Configuration for NuScenes

**File**: `tools/cfgs/nuscenes_models/voxel_rcnn_car_eval.yaml`

This config is for evaluating the CARLA-trained model on NuScenes:
- Same model architecture as training config
- Point to NuScenes validation info files
- `CLASS_NAMES: ['car']`

### Phase 5: Training Workflow

```bash
# Step 1: Generate CARLA info files
cd tools
python create_carla_infos.py --data_path ../data/nusc_like_multi

# Step 2: Train on CARLA data
python train.py --cfg_file cfgs/carla_models/voxel_rcnn_car.yaml

# Step 3: Evaluate on NuScenes validation set
python test.py --cfg_file cfgs/nuscenes_models/voxel_rcnn_car_eval.yaml \
    --ckpt ../output/carla_models/voxel_rcnn_car/default/ckpt/checkpoint_epoch_30.pth
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `tools/create_carla_infos.py` | Generate info files from CARLA JSON metadata |
| `tools/cfgs/dataset_configs/carla_dataset.yaml` | Dataset config for CARLA data |
| `tools/cfgs/carla_models/voxel_rcnn_car.yaml` | VoxelRCNN training config for CARLA |
| `tools/cfgs/nuscenes_models/voxel_rcnn_car_eval.yaml` | Evaluation config for NuScenes |

---

## Technical Considerations

### 1. Coordinate Transformation
The annotation `translation` and `rotation` in CARLA data are in **global frame**. We must:
- Use `ego_pose` to transform to ego frame
- Use `calibrated_sensor` (offset [0,0,2]) to transform to lidar frame
- The rotation quaternion format is [w, x, y, z]

### 2. Box Size Convention
- CARLA annotation `size` is [width, length, height]
- OpenPCDet expects [dx, dy, dz] = [length, width, height]
- Need to swap: `dims = size[[1, 0, 2]]`

### 3. No Velocity Data
- CARLA data doesn't have velocity annotations
- Set `PRED_VELOCITY: False` in config
- gt_boxes will be 7D instead of 9D

### 4. Single Frame (No Sweeps)
- Set `MAX_SWEEPS: 1`
- `sweeps` list in info will be empty
- Point cloud will only contain current frame data

### 5. Domain Gap Considerations
- CARLA simulation has cleaner point clouds than real NuScenes
- Consider adding noise augmentation during training
- Evaluation on NuScenes will test domain transfer capability

### 6. Class Mapping
- CARLA `vehicle.car` → OpenPCDet `car`
- Filter out `human.pedestrian.adult` since we only want vehicle detection
