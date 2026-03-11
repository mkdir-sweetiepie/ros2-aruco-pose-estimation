#!/usr/bin/env python3

# Code taken and readapted from:
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/tree/main

import numpy as np
import cv2
import tf_transformations

from rclpy.impl import rcutils_logger

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from aruco_interfaces.msg import ArucoMarkers

from aruco_pose_estimation.utils import aruco_display


def pose_estimation(rgb_frame: np.array, depth_frame: np.array,
                    aruco_detector: cv2.aruco.ArucoDetector, marker_size: float,
                    matrix_coefficients: np.array, distortion_coefficients: np.array,
                    pose_array: PoseArray, markers: ArucoMarkers) -> list:
    """
    ArUco 마커를 검출하고 3D 포즈를 추정한다.

    position 계산 전략:
      depth 있음 → corners_to_3d() 로 코너 4개 각각 역투영 후 평균
                   (Aruco3DNode 방식 — SolvePnP lateral 편향 없음)
      depth 없음 / depth 실패 → SolvePnP tvec fallback
    orientation 계산:
      항상 SolvePnP quaternion 사용 (depth와 무관하게 정확)
    """

    corners, marker_ids, _ = aruco_detector.detectMarkers(image=rgb_frame)
    frame_processed = rgb_frame
    logger = rcutils_logger.RcutilsLogger(name="aruco_node")

    if len(corners) == 0:
        return frame_processed, pose_array, markers

    logger.debug("Detected {} markers.".format(len(corners)))

    for i, marker_id in enumerate(marker_ids):

        # ── orientation 계산 (SolvePnP) ────────────────────────────────
        # tvec は position には使わないが、2 つの目的で必要:
        #   1) cv2.drawFrameAxes() で座標軸を描画するため
        #   2) rvec → quaternion 変換で orientation を得るため
        tvec, rvec, quat = my_estimatePoseSingleMarkers(
            corners=corners[i],
            marker_size=marker_size,
            camera_matrix=matrix_coefficients,
            distortion=distortion_coefficients
        )

        # 시각화
        frame_processed = aruco_display(corners=corners, ids=marker_ids,
                                        image=frame_processed)
        frame_processed = cv2.drawFrameAxes(
            image=frame_processed,
            cameraMatrix=matrix_coefficients,
            distCoeffs=distortion_coefficients,
            rvec=rvec, tvec=tvec,
            length=0.05, thickness=3
        )

        # ── position 계산 ───────────────────────────────────────────────
        pose = Pose()
        use_depth_position = False

        if depth_frame is not None:
            pts3d = corners_to_3d(
                corners=corners[i],
                depth_image=depth_frame,
                intrinsic_matrix=matrix_coefficients
            )
            if pts3d is not None:
                # 유효한 코너들의 3D 좌표 평균 → 마커 중심
                center = pts3d.mean(axis=0)
                pose.position.x = float(center[0])
                pose.position.y = float(center[1])
                pose.position.z = float(center[2])
                use_depth_position = True

                logger.debug(
                    f"[id={marker_id[0]}] depth center = "
                    f"[{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]"
                )
                logger.debug(
                    f"[id={marker_id[0]}] solvePnP tvec = "
                    f"[{float(tvec[0]):.4f}, {float(tvec[1]):.4f}, {float(tvec[2]):.4f}]"
                )

        if not use_depth_position:
            # depth 없거나 유효 코너 부족 → SolvePnP fallback
            logger.warn(
                f"[id={marker_id[0]}] depth position unavailable, using solvePnP tvec"
            )
            pose.position.x = float(tvec[0])
            pose.position.y = float(tvec[1])
            pose.position.z = float(tvec[2])

        # orientation (항상 SolvePnP quaternion)
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        pose_array.poses.append(pose)
        markers.poses.append(pose)
        markers.marker_ids.append(marker_id[0])

    return frame_processed, pose_array, markers


def corners_to_3d(corners: np.array, depth_image: np.array,
                  intrinsic_matrix: np.array,
                  depth_patch_size: int = 5) -> np.array:
    """
    Aruco3DNode 방식: 마커 코너 4개 각각의 depth를 직접 읽어 역투영.

    기존 depth_to_pointcloud_centroid()와의 차이:
      기존: 마커 내부 모든 픽셀 순회 → pointcloud → 평균
            문제: dtype=np.uint16 버그(음수 오버플로우, 소수점 손실),
                  내부 픽셀 depth 분포 넓을 때 centroid 오차

      수정: 코너 4개 픽셀 주변 patch의 median depth → 역투영
            수식: z = depth_mm / 1000
                  x = (u - cx) * z / fx
                  y = (v - cy) * z / fy
            장점: float32 연산, SolvePnP lateral 편향 없음, 빠름

    depth_patch_size: 각 코너 주변 NxN 패치 크기 (기본 5px)
    return: shape (N, 3) float32, 유효 코너 2개 미만이면 None
    """

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    h, w = depth_image.shape[:2]
    r = max(1, depth_patch_size // 2)

    pts3d = []
    # corners shape: (1, 4, 2) — 순서: 좌상, 우상, 우하, 좌하
    for corner in corners[0]:
        u, v = int(corner[0]), int(corner[1])

        # 이미지 경계 밖이면 스킵
        if not (0 <= u < w and 0 <= v < h):
            continue

        # 패치 범위 (경계 클리핑)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        patch = depth_image[y0:y1, x0:x1]

        # 0은 depth 측정 실패를 의미하므로 제외
        valid = patch[patch > 0]
        if valid.size == 0:
            continue

        # median으로 노이즈에 robust하게 depth 추출
        depth_m = float(np.median(valid)) / 1000.0  # mm → m

        # D405 유효 측정 범위: 7cm ~ 3m
        if not (0.07 <= depth_m <= 3.0):
            continue

        # 핀홀 카메라 역투영
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m

        pts3d.append([x, y, z])

    if len(pts3d) < 2:
        return None

    return np.array(pts3d, dtype=np.float32)


def my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix,
                                  distortion) -> tuple:
    """
    SolvePnP로 rvec, tvec, quaternion 계산.

    position 계산에는 tvec을 더 이상 쓰지 않지만,
    drawFrameAxes 시각화와 orientation(quaternion) 추출에 필요.

    SOLVEPNP_IPPE_SQUARE: 정사각형 마커에 최적화된 알고리즘.
    rvec → Rodrigues → 회전행렬 → quaternion 변환.
    """

    marker_points = np.array([
        [-marker_size / 2.0,  marker_size / 2.0, 0],
        [ marker_size / 2.0,  marker_size / 2.0, 0],
        [ marker_size / 2.0, -marker_size / 2.0, 0],
        [-marker_size / 2.0, -marker_size / 2.0, 0],
    ], dtype=np.float32)

    _, rvec, tvec = cv2.solvePnP(
        objectPoints=marker_points,
        imagePoints=corners,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    rot, _ = cv2.Rodrigues(rvec)
    rot_matrix = np.eye(4, dtype=np.float32)
    rot_matrix[0:3, 0:3] = rot

    quaternion = tf_transformations.quaternion_from_matrix(rot_matrix)
    quaternion = quaternion / np.linalg.norm(quaternion)

    return tvec, rvec, quaternion

# ── 제거된 함수 ──────────────────────────────────────────────────────────────
# depth_to_pointcloud_centroid() → corners_to_3d()로 교체
#   이유: dtype=np.uint16 버그, 내부 픽셀 순회 방식의 오차 및 성능 문제
# is_pixel_in_polygon() → depth_to_pointcloud_centroid()의 헬퍼였으므로 함께 제거
# ────────────────────────────────────────────────────────────────────────────