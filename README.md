# ROS2 ArUco Pose Estimation

ROS2 wrapper for ArUco marker detection and pose estimation using OpenCV. Supports RGB-only and RGB+Depth pose estimation modes. Compatible with ROS2 Humble and Iron.

Designed for dual-camera setups (left/right) with Intel RealSense D405. Works with any camera that provides standard ROS2 image topics.

> Originally developed by Simone Giampà at Politecnico di Milano (2024).

## Packages

| Package | Description |
|---------|-------------|
| `aruco_pose_estimation` | Main node for ArUco marker detection and pose estimation |
| `aruco_interfaces` | Custom message definition (`ArucoMarkers.msg`) |

## Installation

### Dependencies

```bash
pip3 install opencv-python opencv-contrib-python transforms3d

# ROS2 Humble
sudo apt install ros-humble-tf-transformations
```

### Build

```bash
cd ~/colcon_ws
colcon build --symlink-install --packages-up-to aruco_pose_estimation
source install/setup.bash
```

## Usage

### Launch

```bash
# Left camera
ros2 launch aruco_pose_estimation aruco_pose_estimation.launch.py camera_side:=left

# Right camera
ros2 launch aruco_pose_estimation aruco_pose_estimation.launch.py camera_side:=right

# With RViz visualization
ros2 launch aruco_pose_estimation aruco_pose_estimation.launch.py camera_side:=left launch_rviz:=true
```

### Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `camera_side` | `left` | Camera to use (`left` or `right`) |
| `launch_rviz` | `false` | Launch RViz2 with preset configuration |

## Configuration

Parameters are defined per camera side in YAML config files:

- `config/aruco_parameters_left.yaml` — Right camera (D405) configuration
- `config/aruco_parameters_right.yaml` — Left camera (D405) configuration

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `marker_size` | `0.12` | Marker size in meters |
| `aruco_dictionary_id` | `DICT_4X4_50` | ArUco dictionary type |
| `use_depth_input` | `true` | Use depth image for pose estimation |
| `image_topic` | Camera-dependent | RGB image topic |
| `depth_image_topic` | Camera-dependent | Depth image topic |
| `camera_info_topic` | Camera-dependent | Camera info topic |
| `camera_frame` | Camera-dependent | Camera optical frame |
| `detected_markers_topic` | `/l/aruco/markers` or `/r/aruco/markers` | Detected markers output topic |
| `markers_visualization_topic` | `/l/aruco/poses` or `/r/aruco/poses` | PoseArray output topic |
| `output_image_topic` | `/l/aruco/image` or `/r/aruco/image` | Annotated image output topic |

### Supported ArUco Dictionaries

`DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_4X4_1000`,
`DICT_5X5_50`, `DICT_5X5_100`, `DICT_5X5_250`, `DICT_5X5_1000`,
`DICT_6X6_50`, `DICT_6X6_100`, `DICT_6X6_250`, `DICT_6X6_1000`,
`DICT_7X7_50`, `DICT_7X7_100`, `DICT_7X7_250`, `DICT_7X7_1000`,
`DICT_ARUCO_ORIGINAL`,
`DICT_APRILTAG_16h5`, `DICT_APRILTAG_25h9`, `DICT_APRILTAG_36h10`, `DICT_APRILTAG_36h11`

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `image_topic` | `sensor_msgs/Image` | RGB image input |
| `depth_image_topic` | `sensor_msgs/Image` | Depth image input (when `use_depth_input: true`) |
| `camera_info_topic` | `sensor_msgs/CameraInfo` | Camera intrinsic and distortion parameters |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `markers_visualization_topic` | `geometry_msgs/PoseArray` | Poses of detected markers (for RViz) |
| `detected_markers_topic` | `aruco_interfaces/ArucoMarkers` | Marker IDs with corresponding poses |
| `output_image_topic` | `sensor_msgs/Image` | Annotated image with marker bounding boxes |

## Pose Estimation Method

The node uses a hybrid approach for accurate pose estimation:

**Position:** When depth input is enabled, marker corner pixels are backprojected to 3D using the depth image and pinhole camera model.

**Orientation:** Always estimated via `cv2.solvePnP` with `SOLVEPNP_IPPE_SQUARE` (optimized for square planar markers), converting the rotation vector to a quaternion.

## Custom Message

### `aruco_interfaces/ArucoMarkers`

```
std_msgs/Header header
int64[] marker_ids
geometry_msgs/Pose[] poses
```

## License

Apache License 2.0
