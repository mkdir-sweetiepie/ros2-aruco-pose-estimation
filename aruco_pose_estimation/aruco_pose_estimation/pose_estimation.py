#!/usr/bin/env python3

# 참고 코드 출처:
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/tree/main

import numpy as np
import cv2
import tf_transformations

from rclpy.impl import rcutils_logger

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from aruco_interfaces.msg import ArucoMarkers

from aruco_pose_estimation.utils import aruco_display


# ═══════════════════════════════════════════════════════════════════════════════
# 전체 설계 철학
# ───────────────────────────────────────────────────────────────────────────────
#
#  [position 결정 흐름]
#
#   RGB 프레임에서 마커 코너 검출
#          │
#          ▼
#   depth_frame 있음? ──No──→ SolvePnP tvec 사용 (fallback)
#          │Yes
#          ▼
#   corners_to_3d() : 코너 4개 각각 depth 읽어 역투영
#          │
#   유효 코너 2개 이상? ──No──→ SolvePnP tvec fallback
#          │Yes
#          ▼
#   4개 3D 좌표 평균 → 마커 중심 (position)
#
#  [orientation 결정 흐름]
#
#   SolvePnP → rvec → Rodrigues → 회전행렬 → quaternion
#   (depth 유무와 무관하게 항상 이 경로 사용)
#
#  왜 position에 SolvePnP tvec을 안 쓰나?
#    - SolvePnP는 마커 평면을 가정한 최적화이므로
#      카메라와 마커가 비스듬할 때 lateral(x,y) 편향이 생김
#    - depth 역투영은 실제 측정값 기반이라 편향이 없음
# ═══════════════════════════════════════════════════════════════════════════════


def pose_estimation(rgb_frame: np.array, depth_frame: np.array,
                    aruco_detector: cv2.aruco.ArucoDetector, marker_size: float,
                    matrix_coefficients: np.array, distortion_coefficients: np.array,
                    pose_array: PoseArray, markers: ArucoMarkers) -> list:
    """
    RGB + Depth 프레임에서 ArUco 마커를 검출하고 3D 포즈를 추정한다.

    Args:
        rgb_frame            : 컬러 이미지 (HxWx3, uint8)
        depth_frame          : 깊이 이미지 (HxW, uint16, 단위 mm). 없으면 None
        aruco_detector       : cv2.aruco.ArucoDetector 인스턴스
        marker_size          : 마커 실물 한 변 길이 (단위: m)
        matrix_coefficients  : 카메라 내부 파라미터 행렬 (3x3)
        distortion_coefficients: 왜곡 계수 벡터
        pose_array           : 결과를 누적할 PoseArray ROS 메시지
        markers              : 결과를 누적할 ArucoMarkers ROS 메시지

    Returns:
        (frame_processed, pose_array, markers)
        frame_processed: 마커/좌표축이 그려진 시각화 이미지
    """

    # ── 1. 마커 검출 ─────────────────────────────────────────────────────────
    # detectMarkers(): RGB 이미지에서 ArUco 마커를 찾아
    #   corners  : 검출된 마커별 코너 좌표 리스트  [N × (1,4,2)]
    #   marker_ids: 각 마커의 ID 배열              [N × 1]
    #   _        : rejected 후보 (사용 안 함)
    corners, marker_ids, _ = aruco_detector.detectMarkers(image=rgb_frame)
    frame_processed = rgb_frame
    logger = rcutils_logger.RcutilsLogger(name="aruco_node")

    # 검출된 마커가 없으면 그냥 반환
    if len(corners) == 0:
        return frame_processed, pose_array, markers

    logger.debug("Detected {} markers.".format(len(corners)))

    # ── 2. 검출된 마커 하나씩 처리 ───────────────────────────────────────────
    for i, marker_id in enumerate(marker_ids):

        # ── 2-1. orientation 계산 (SolvePnP) ─────────────────────────────
        # tvec: 카메라 좌표계에서 마커 중심까지의 벡터 (3,1)
        #       ← position에는 미사용, drawFrameAxes 시각화에만 쓰임
        # rvec: 회전 벡터 (Rodrigues 표현)   (3,1)
        # quat: [x, y, z, w] quaternion
        tvec, rvec, quat = my_estimatePoseSingleMarkers(
            corners=corners[i],
            marker_size=marker_size,
            camera_matrix=matrix_coefficients,
            distortion=distortion_coefficients
        )

        # ── 2-2. 시각화 ───────────────────────────────────────────────────
        # aruco_display : 마커 외곽선 + ID 텍스트 오버레이
        frame_processed = aruco_display(corners=corners, ids=marker_ids,
                                        image=frame_processed)
        # drawFrameAxes  : rvec/tvec을 이용해 XYZ 좌표축 그리기
        #   length=0.05 → 5cm 길이 축
        frame_processed = cv2.drawFrameAxes(
            image=frame_processed,
            cameraMatrix=matrix_coefficients,
            distCoeffs=distortion_coefficients,
            rvec=rvec, tvec=tvec,
            length=0.05, thickness=3
        )

        # ── 2-3. position 계산 ───────────────────────────────────────────
        pose = Pose()
        use_depth_position = False  # depth 경로 성공 여부 플래그

        if depth_frame is not None:
            # corners_to_3d(): depth 이미지 + 코너 픽셀 좌표 → 3D 포인트 배열
            pts3d = corners_to_3d(
                corners=corners[i],
                depth_image=depth_frame,
                intrinsic_matrix=matrix_coefficients
            )
            if pts3d is not None:
                # 유효한 코너들의 3D 좌표를 평균 → 마커 중심 위치
                # pts3d shape: (N, 3),  N ≤ 4
                center = pts3d.mean(axis=0)
                pose.position.x = float(center[0])
                pose.position.y = float(center[1])
                pose.position.z = float(center[2])
                use_depth_position = True  # depth 경로 성공

                logger.debug(
                    f"[id={marker_id[0]}] depth center = "
                    f"[{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]"
                )
                logger.debug(
                    f"[id={marker_id[0]}] solvePnP tvec = "
                    f"[{float(tvec[0]):.4f}, {float(tvec[1]):.4f}, {float(tvec[2]):.4f}]"
                )

        if not use_depth_position:
            # depth가 없거나, 유효 코너가 2개 미만이면 SolvePnP tvec으로 fallback
            logger.warn(
                f"[id={marker_id[0]}] depth position unavailable, using solvePnP tvec"
            )
            pose.position.x = float(tvec[0])
            pose.position.y = float(tvec[1])
            pose.position.z = float(tvec[2])

        # ── 2-4. orientation 채우기 (항상 SolvePnP quaternion) ───────────
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        # ── 2-5. ROS 메시지에 누적 ────────────────────────────────────────
        pose_array.poses.append(pose)
        markers.poses.append(pose)
        markers.marker_ids.append(marker_id[0])

    return frame_processed, pose_array, markers


# ───────────────────────────────────────────────────────────────────────────────
def corners_to_3d(corners: np.array, depth_image: np.array,
                  intrinsic_matrix: np.array,
                  depth_patch_size: int = 5) -> np.array:
    """
    마커 코너 4개의 픽셀 좌표 + depth 이미지 → 카메라 좌표계 3D 좌표 배열.

    ┌─────────────────────────────────────────────────────────────┐
    │  핀홀 카메라 역투영 공식                                      │
    │                                                              │
    │   z = depth_mm / 1000          (mm → m)                     │
    │   x = (u - cx) * z / fx                                     │
    │   y = (v - cy) * z / fy                                     │
    │                                                              │
    │   (u, v) : 픽셀 좌표                                         │
    │   (fx,fy): 초점 거리  (intrinsic_matrix 대각 원소)            │
    │   (cx,cy): 주점 (principal point)                            │
    └─────────────────────────────────────────────────────────────┘

    구 버전(depth_to_pointcloud_centroid)과의 차이
    ─────────────────────────────────────────────
    구버전 방식  : 마커 내부 모든 픽셀 순회 → point cloud → 평균
    문제점
      ① dtype=np.uint16 연산 중 음수 오버플로우 → 잘못된 depth 값
      ② 소수점 손실 (uint16은 정수형)
      ③ 마커 내부 픽셀 depth 분포가 넓을 때 centroid 오차 큼
      ④ 픽셀 수가 많아 느림

    현재 방식  : 코너 4개 픽셀의 NxN 패치 median depth 로만 계산
    장점
      ① float32 연산 → 오버플로우·소수점 손실 없음
      ② 코너 4개만 보므로 빠름
      ③ patch median으로 센서 노이즈에 robust
      ④ SolvePnP의 lateral 편향 없음 (실측 depth 사용)

    Args:
        corners          : (1, 4, 2) 코너 픽셀 좌표  순서: 좌상/우상/우하/좌하
        depth_image      : (H, W) uint16, 단위 mm
        intrinsic_matrix : (3, 3) 카메라 내부 파라미터
        depth_patch_size : 코너 주변 NxN 패치 크기 (default 5px)

    Returns:
        np.array shape (N, 3) float32  (N = 유효 코너 수, N ≤ 4)
        유효 코너가 2개 미만이면 None  (position 계산 불가 기준)
    """

    # intrinsic 행렬에서 focal length + principal point 추출
    fx = intrinsic_matrix[0, 0]   # x 방향 초점 거리 (픽셀 단위)
    fy = intrinsic_matrix[1, 1]   # y 방향 초점 거리
    cx = intrinsic_matrix[0, 2]   # 이미지 중심 x (주점)
    cy = intrinsic_matrix[1, 2]   # 이미지 중심 y

    h, w = depth_image.shape[:2]
    r = max(1, depth_patch_size // 2)   # 패치 반지름 (최소 1px)

    pts3d = []

    # corners shape: (1, 4, 2) → corners[0]은 4개 (u,v) 배열
    for corner in corners[0]:
        u, v = int(corner[0]), int(corner[1])

        # 이미지 경계 밖 코너 스킵
        if not (0 <= u < w and 0 <= v < h):
            continue

        # 패치 범위 계산 (이미지 경계 클리핑)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        patch = depth_image[y0:y1, x0:x1]  # NxN 서브이미지

        # depth=0 은 측정 실패 픽셀이므로 제거
        valid = patch[patch > 0]
        if valid.size == 0:
            continue

        # median: 이상치(물체 경계 등)에 덜 민감
        depth_m = float(np.median(valid)) / 1000.0  # mm → m

        # Intel RealSense D405 유효 측정 범위 필터 (7cm ~ 3m)
        if not (0.07 <= depth_m <= 3.0):
            continue

        # 핀홀 역투영 → 카메라 좌표계 3D 포인트
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m

        pts3d.append([x, y, z])

    # 2개 미만이면 평균이 마커 중심을 대표하지 못한다고 판단 → None
    if len(pts3d) < 2:
        return None

    return np.array(pts3d, dtype=np.float32)


# ───────────────────────────────────────────────────────────────────────────────
def my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix,
                                  distortion) -> tuple:
    """
    단일 마커에 대해 SolvePnP로 rvec, tvec, quaternion을 계산한다.

    사용 알고리즘: SOLVEPNP_IPPE_SQUARE
      - 정사각형 마커에 특화된 해석적(closed-form) 알고리즘
      - 일반 SOLVEPNP_ITERATIVE보다 속도 빠르고 수치 안정성 높음

    좌표 변환 흐름:
      corners(2D) + marker_points(3D) → SolvePnP
        → rvec (Rodrigues 회전벡터)
        → cv2.Rodrigues() → 3x3 회전행렬
        → 4x4 동차 행렬
        → tf_transformations.quaternion_from_matrix()
        → [x, y, z, w] 정규화된 quaternion

    Args:
        corners      : (1, 4, 2) 검출된 마커 코너 픽셀 좌표
        marker_size  : 마커 실물 한 변 길이 (m)
        camera_matrix: (3, 3) 카메라 내부 파라미터
        distortion   : 왜곡 계수

    Returns:
        tvec     : (3,1) 카메라→마커 변환 벡터  ← position에는 미사용
        rvec     : (3,1) Rodrigues 회전벡터     ← drawFrameAxes에 필요
        quaternion: (4,)  [x, y, z, w] 정규화된 회전 quaternion
    """

    # 마커 3D 좌표 정의 (마커 평면 = z=0, 마커 중심 = 원점)
    #   순서: 좌상(-,+) / 우상(+,+) / 우하(+,-) / 좌하(-,-)
    #   detectMarkers()의 코너 순서와 1:1 대응
    marker_points = np.array([
        [-marker_size / 2.0,  marker_size / 2.0, 0],
        [ marker_size / 2.0,  marker_size / 2.0, 0],
        [ marker_size / 2.0, -marker_size / 2.0, 0],
        [-marker_size / 2.0, -marker_size / 2.0, 0],
    ], dtype=np.float32)

    # PnP 문제: 3D–2D 대응점으로 카메라 외부 파라미터(R,t) 추정
    _, rvec, tvec = cv2.solvePnP(
        objectPoints=marker_points,
        imagePoints=corners,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    # shape 정리: (3,) → (3,1) (drawFrameAxes API 요구사항)
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    # Rodrigues 회전벡터 → 3x3 회전행렬
    rot, _ = cv2.Rodrigues(rvec)

    # 4x4 동차 변환행렬 구성 (quaternion_from_matrix() 입력 형식)
    rot_matrix = np.eye(4, dtype=np.float32)
    rot_matrix[0:3, 0:3] = rot

    # 회전행렬 → quaternion [x, y, z, w]
    quaternion = tf_transformations.quaternion_from_matrix(rot_matrix)
    # 수치 오차 보정: 단위 quaternion으로 정규화
    quaternion = quaternion / np.linalg.norm(quaternion)

    return tvec, rvec, quaternion


# ── 제거된 함수 (변경 이력) ───────────────────────────────────────────────────
#
#  depth_to_pointcloud_centroid()
#    역할: 마커 내부 모든 픽셀을 순회해 point cloud를 만들고 centroid 반환
#    제거 이유:
#      ① dtype=np.uint16 덧셈/뺄셈 중 음수 오버플로우 버그
#      ② 정수 연산으로 소수점 손실
#      ③ 마커 경계에서 depth 불연속 시 centroid 오차 큼
#      ④ 내부 픽셀 전체 순회 → 느림
#    대체: corners_to_3d()  (코너 4개 patch median, float32 연산)
#
#  is_pixel_in_polygon()
#    역할: depth_to_pointcloud_centroid()의 내부 픽셀 필터 헬퍼
#    제거 이유: 상위 함수 제거로 함께 불필요
# ─────────────────────────────────────────────────────────────────────────────