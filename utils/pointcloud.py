# -*- coding: utf-8 -*-
# @Author  : LG

import logging
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import laspy
from plyfile import PlyData
from typing import Tuple


logger = logging.getLogger(__name__)


def las_read(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = laspy.read(file_path)
    vertices = np.vstack((data.x, data.y, data.z)).transpose()
    try:
        rgb = np.vstack((data.red, data.green, data.blue)).transpose()
        rgb = rgb >> 8
        rgb = rgb / 255
    except:
        rgb = np.zeros(vertices.shape)
        print('LasData object has no attribute [red, green, blue], {}'.format(data.point_format))
    vertices = vertices.astype(np.float32)
    rgb = rgb.astype(np.float32)
    xmin, ymin, zmin = min(vertices[:, 0]), min(vertices[:, 1]), min(vertices[:, 2])
    xmax, ymax, zmax = max(vertices[:, 0]), max(vertices[:, 1]), max(vertices[:, 2])
    vertices -= (xmin, ymin, zmin)

    offset = data.header.offset + np.array([xmin, ymin, zmin])
    size = np.array((xmax - xmin, ymax - ymin, zmax - zmin))
    return vertices, rgb, size, offset


def ply_read(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ply_data = PlyData.read(file_path)
    if 'vertex' not in ply_data:
        return np.array([]), np.array([]), np.array([0, 0, 0]), np.array([0, 0, 0])

    vertices = np.vstack((ply_data['vertex']['x'],
                          ply_data['vertex']['y'],
                          ply_data['vertex']['z'])).transpose()
    if 'red' in ply_data['vertex']:
        colors = np.vstack((ply_data['vertex']['red'],
                            ply_data['vertex']['green'],
                            ply_data['vertex']['blue'])).transpose()
    else:
        colors = np.ones(vertices.shape)

    vertices = vertices.astype(np.float32)
    xmin, ymin, zmin = min(vertices[:, 0]), min(vertices[:, 1]), min(vertices[:, 2])
    xmax, ymax, zmax = max(vertices[:, 0]), max(vertices[:, 1]), max(vertices[:, 2])
    size = np.array((xmax - xmin, ymax - ymin, zmax - zmin))
    offset = np.array([xmin, ymin, zmin])
    vertices -= offset
    colors = colors.astype(np.float32)
    colors = colors / 255
    return vertices, colors, size, offset


def txt_read(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    读取以ASCII文本存储的点云文件。
    支持如下布局：
      - XYZ（至少3列）
      - XYZRGB（至少6列，RGB可为0-1或0-255）
    额外列会被忽略。
    """
    logger.debug("txt_read | path=%s", file_path)
    datas = np.loadtxt(file_path, dtype=np.float32)
    if datas.ndim == 1:
        datas = datas.reshape(1, -1)
    if datas.shape[1] < 3:
        raise ValueError(f"Point cloud file '{file_path}' must contain at least 3 columns for XYZ.")
    vertices = datas[:, :3].astype(np.float32, copy=False)

    rgb: np.ndarray
    if datas.shape[1] >= 6:
        rgb = datas[:, 3:6].astype(np.float32, copy=False)
        if rgb.size == 0:
            logger.debug("txt_read | rgb columns detected but empty -> default white")
            rgb = np.ones_like(vertices, dtype=np.float32)
        else:
            max_val = float(np.max(rgb))
            min_val = float(np.min(rgb))
            logger.debug("txt_read | raw rgb range min=%.3f max=%.3f", min_val, max_val)
            if max_val > 1.0001 or min_val < -0.0001:
                rgb = np.clip(rgb, 0.0, 255.0) / 255.0
                logger.debug("txt_read | rgb normalized from 0-255 to 0-1")
            else:
                rgb = np.clip(rgb, 0.0, 1.0)
                logger.debug("txt_read | rgb already in 0-1 range")
    else:
        rgb = np.ones_like(vertices, dtype=np.float32)
        logger.debug("txt_read | only XYZ columns -> default white")

    xmin, ymin, zmin = float(np.min(vertices[:, 0])), float(np.min(vertices[:, 1])), float(np.min(vertices[:, 2]))
    xmax, ymax, zmax = float(np.max(vertices[:, 0])), float(np.max(vertices[:, 1])), float(np.max(vertices[:, 2]))
    vertices -= np.array([xmin, ymin, zmin], dtype=np.float32)
    logger.debug(
        "txt_read | total_points=%d bbox_min=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f)",
        vertices.shape[0], xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin
    )
    offset = np.array([xmin, ymin, zmin], dtype=np.float32)
    size = np.array((xmax - xmin, ymax - ymin, zmax - zmin), dtype=np.float32)
    return vertices, rgb.astype(np.float32, copy=False), size, offset


class PointCloud:
    def __init__(self, file_path:str, xyz, rgb, size, offset):
        self.file_path:str = file_path
        self.xyz:np.ndarray = xyz if xyz.dtype == np.float32 else xyz.astype(np.float32)
        self.offset:np.ndarray = offset
        self.size:np.ndarray = size
        self.num_point = self.xyz.shape[0]
        self.rgb:np.ndarray = rgb if rgb.dtype == np.float32 else rgb.astype(np.float32)

    def __str__(self):
        return "<PointCloud num_point: {} | size: ({:.2f}, {:.2f}, {:.2f}) | offset: ({:.2f}, {:.2f}, {:.2f}) >".format(
            self.num_point, self.size[0], self.size[1], self.size[2], self.offset[0], self.offset[1], self.offset[2])


class PointCloudReadThread(QThread):
    message = pyqtSignal(str, int)
    tag = pyqtSignal(bool)

    def __init__(self):
        super(PointCloudReadThread, self).__init__()
        self.file_path = None
        self.callback = None
        self.pointcloud = None

    def run(self):
        self.message.emit("Open file | Loading ...", 10000000)    # 一直显示
        try:
            self.pointcloud = self.read(self.file_path)
            self.message.emit("Open file | Load point cloud finished.", 1000)
            self.tag.emit(True)
        except Exception as e:
            self.pointcloud = None
            self.message.emit(f"Open file | Error: {e}", 5000)
            self.tag.emit(False)

    @staticmethod
    def read(file_path:str):
        lower_path = file_path.lower()
        if lower_path.endswith('.las'):
            xyz, rgb, size, offset = las_read(file_path)
        elif lower_path.endswith('.ply'):
            xyz, rgb, size, offset = ply_read(file_path)
        elif lower_path.endswith('.txt') or lower_path.endswith('.xyz') or lower_path.endswith('.xyzrgb'):
            xyz, rgb, size, offset = txt_read(file_path)
        else:
            return None
        pointcloud = PointCloud(file_path, xyz, rgb, size, offset)
        return pointcloud

    def __del__(self):
        self.wait()

    def set_file_path(self, file_path):
        self.file_path = file_path

    def set_callback(self, callback):
        self.callback = callback
