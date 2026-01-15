import os, json, argparse
import numpy as np
import open3d as o3d
import transforms3d.quaternions

# --- Math Helpers ---
def quat_wxyz_to_R(q):
    return transforms3d.quaternions.quat2mat(np.array(q, dtype=np.float64))

def make_T(t, q):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = quat_wxyz_to_R(q)
    T[:3,3] = np.array(t, dtype=np.float64)
    return T

def inv_T(T):
    R = T[:3,:3]
    t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ t
    return Ti

def load_table(root, version, name):
    p = os.path.join(root, version, name)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Visualizer Class ---
class NuscVisualizer:
    def __init__(self, root, version, lidar_name, start_idx=0):
        self.root = root
        self.version = version
        self.lidar_name = lidar_name
        self.idx = start_idx
        
        # Load metadata
        print(f"Loading tables from {os.path.join(root, version)}...")
        self.samples = load_table(root, version, "sample.json")
        sample_data = load_table(root, version, "sample_data.json")
        ego_pose = load_table(root, version, "ego_pose.json")
        calib = load_table(root, version, "calibrated_sensor.json")
        self.anns = load_table(root, version, "sample_annotation.json")

        self.sd_by_token = {x["token"]: x for x in sample_data}
        self.ego_by_token = {x["token"]: x for x in ego_pose}
        self.calib_by_token = {x["token"]: x for x in calib}

        # State storage
        self.pcd = o3d.geometry.PointCloud()
        self.box_geoms = []
        self.first_render = True

    def load_frame_data(self):
        """Loads data for self.idx into self.pcd and returns new box geometries"""
        # Bounds check
        if self.idx < 0: self.idx = 0
        if self.idx >= len(self.samples): self.idx = len(self.samples) - 1

        s = self.samples[self.idx]
        sd_token = s["data"][self.lidar_name]
        sd = self.sd_by_token[sd_token]
        
        ego = self.ego_by_token[sd["ego_pose_token"]]
        cs = self.calib_by_token[sd["calibrated_sensor_token"]]

        T_ego_global = make_T(ego["translation"], ego["rotation"])
        T_lidar_ego = make_T(cs["translation"], cs["rotation"])
        T_lidar_global = T_ego_global @ T_lidar_ego
        T_global_lidar = inv_T(T_lidar_global)

        # Load LIDAR
        bin_path = os.path.join(self.root, sd["filename"])
        raw_data = np.fromfile(bin_path, dtype=np.float32)
        
        # FIX: Check if divisible by 5 (NuScenes standard) or 4 (Basic)
        if raw_data.size % 5 == 0:
            pts = raw_data.reshape(-1, 5)
        elif raw_data.size % 4 == 0:
            pts = raw_data.reshape(-1, 4)
        else:
            print(f"Error: Point cloud size {raw_data.size} is not divisible by 4 or 5.")
            return [], "Error loading frame"
        
        xyz = pts[:, :3]
        intensity = pts[:, 3:4]

        # Reset PCD data
        self.pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        if intensity.size > 0:
            i = intensity
            i = (i - i.min()) / (i.max() - i.min() + 1e-6)
            colors = np.repeat(i, 3, axis=1)
            self.pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        # Generate Boxes
        new_boxes = []
        current_anns = [a for a in self.anns if a["sample_token"] == s["token"]]
        for a in current_anns:
            T_box_global = make_T(a["translation"], a["rotation"])
            T_box_lidar = T_global_lidar @ T_box_global
            
            w, l, h = a["size"]
            # Open3D extent is x,y,z relative to box rotation
            extent = np.array([l, w, h], dtype=np.float64) 
            
            obb = o3d.geometry.OrientedBoundingBox(
                center=T_box_lidar[:3,3], 
                R=T_box_lidar[:3,:3], 
                extent=extent
            )
            obb.color = (1.0, 0.0, 0.0)
            new_boxes.append(obb)
            
        return new_boxes, f"Frame {self.idx}"

    def update_renderer(self, vis):
        """Clear old geometry, add new geometry"""
        # 1. Remove old boxes
        for b in self.box_geoms:
            vis.remove_geometry(b, reset_bounding_box=False)
        
        # 2. Remove old PCD
        vis.remove_geometry(self.pcd, reset_bounding_box=False)

        # 3. Load new data
        new_boxes, title = self.load_frame_data()
        self.box_geoms = new_boxes

        # 4. Add new PCD and Boxes
        vis.add_geometry(self.pcd, reset_bounding_box=False)
        for b in self.box_geoms:
            vis.add_geometry(b, reset_bounding_box=False)

        # 5. Handle Camera Reset (Only on first frame)
        if self.first_render:
            vis.reset_view_point(True)
            self.first_render = False
        
        print(title)

    # Key Callbacks (Robust signature)
    def on_right(self, vis, action=None, mods=None):
        # Compatible with both callback styles:
        # Style 1: f(vis)
        # Style 2: f(vis, action, mods) -> check action==1 (KeyDown)
        if action is not None and action != 1:
            return False
            
        self.idx += 1
        self.update_renderer(vis)
        return True

    def on_left(self, vis, action=None, mods=None):
        if action is not None and action != 1:
            return False

        self.idx -= 1
        self.update_renderer(vis)
        return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", default="v1.0-nusc_like")
    ap.add_argument("--frame_idx", type=int, default=0)
    ap.add_argument("--lidar", default="LIDAR_TOP")
    args = ap.parse_args()

    controller = NuscVisualizer(args.root, args.version, args.lidar, args.frame_idx)

    # Setup Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1280, height=720)

    # Register Keys (GLFW Key Codes: 262=Right, 263=Left)
    vis.register_key_callback(262, controller.on_right)
    vis.register_key_callback(263, controller.on_left)

    # Initial Render
    controller.update_renderer(vis)

    print("\nControls:\n  [Right Arrow]: Next Frame\n  [Left Arrow]:  Prev Frame\n  [Q]:           Quit")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()