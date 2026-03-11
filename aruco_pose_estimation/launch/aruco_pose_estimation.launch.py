# ROS2 imports
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare

import os
from ament_index_python.packages import get_package_share_directory
import yaml


def generate_launch_description():

    # --- camera_side argument ---
    camera_side_arg = DeclareLaunchArgument(
        name='camera_side',
        default_value='left',
        description='Camera side to use: left or right',
        choices=['left', 'right']
    )

    launch_rviz_arg = DeclareLaunchArgument(
        name='launch_rviz',
        default_value='false',
        description='Whether to launch RViz2',
        choices=['true', 'false']
    )

    # We need to resolve camera_side at parse time to load the correct YAML,
    # so we use OpaqueFunction for deferred evaluation.
    from launch.actions import OpaqueFunction

    def launch_setup(context):
        camera_side = LaunchConfiguration('camera_side').perform(context)
        launch_rviz = LaunchConfiguration('launch_rviz').perform(context)

        # Load the corresponding YAML config
        aruco_params_file = os.path.join(
            get_package_share_directory('aruco_pose_estimation'),
            'config',
            f'aruco_parameters_{camera_side}.yaml'
        )

        with open(aruco_params_file, 'r') as file:
            config = yaml.safe_load(file)

        config = config["/aruco_node"]["ros__parameters"]

        # ArUco node with side-specific name
        aruco_node = Node(
            package='aruco_pose_estimation',
            executable='aruco_node.py',
            name=f'aruco_node_{camera_side}',
            parameters=[{
                "marker_size": config['marker_size'],
                "aruco_dictionary_id": config['aruco_dictionary_id'],
                "image_topic": config['image_topic'],
                "use_depth_input": config['use_depth_input'],
                "depth_image_topic": config['depth_image_topic'],
                "camera_info_topic": config['camera_info_topic'],
                "camera_frame": config['camera_frame'],
                "detected_markers_topic": config['detected_markers_topic'],
                "markers_visualization_topic": config['markers_visualization_topic'],
                "output_image_topic": config['output_image_topic'],
            }],
            output='screen',
            emulate_tty=True
        )

        nodes = [aruco_node]

        # Optionally launch RViz2
        if launch_rviz == 'true':
            rviz_file = PathJoinSubstitution([
                FindPackageShare('aruco_pose_estimation'),
                'rviz',
                'cam_detect.rviz'
            ])

            rviz2_node = Node(
                package='rviz2',
                executable='rviz2',
                arguments=['-d', rviz_file]
            )
            nodes.append(rviz2_node)

        return nodes

    return LaunchDescription([
        camera_side_arg,
        launch_rviz_arg,
        OpaqueFunction(function=launch_setup),
    ])
