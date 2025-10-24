# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5.QtCore import QThread, pyqtSignal
# removed: multiprocessing.Pool (caused PyInstaller child process re-exec on Windows)
import numpy as np


class ElevationColorThread(QThread):
    message = pyqtSignal(str, int)
    tag = pyqtSignal(bool)

    def __init__(self):
        super(ElevationColorThread, self).__init__()
        self.vertices:np.ndarray = None
        self.callback = None
        self.elevation_color = None

    def run(self):
        self.message.emit("Calculate elevation | Calculating ...", 10000000)  # 一直显示
        try:
            self.elevation_color = self.cal_color_elevation(self.vertices)
            self.message.emit("Calculate elevation | Calculate finished.", 1000)
            self.tag.emit(True)
        except Exception as e:
            self.elevation_color = None
            self.message.emit(f"Calculate elevation | Error: {e}", 5000)
            self.tag.emit(False)

    @staticmethod
    def cal_color_elevation(vertices):
        """
        计算高差颜色
        :return:
        """
        if vertices is None:
            return
        zs = vertices[:, 2]
        zmin, zmax = min(zs), max(zs)
        colors = np.zeros(vertices.shape, dtype=np.float32)

        split = (zmax - zmin) / 4 * 3 + zmin
        mask = zs >= split
        colors[mask, 0] = 1
        colors[mask, 1] = 1 - (zs[mask] - split) / ((zmax - zmin) / 4)

        split = (zmax - zmin) / 4 * 2 + zmin
        mask = (zs < (zmax - zmin) / 4 * 3 + zmin) & (zs >= (zmax - zmin) / 4 * 2 + zmin)
        colors[mask, 0] = (zs[mask] - split) / ((zmax - zmin) / 4)
        colors[mask, 1] = 1

        split = (zmax - zmin) / 4 * 1 + zmin
        mask = (zs < (zmax - zmin) / 4 * 2 + zmin) & (zs >= (zmax - zmin) / 4 * 1 + zmin)
        colors[mask, 1] = 1
        colors[mask, 2] = 1 - (zs[mask] - split) / ((zmax - zmin) / 4)

        split = (zmax - zmin) / 4 * 0 + zmin
        mask = (zs < (zmax - zmin) / 4 * 1 + zmin)
        colors[mask, 1] = (zs[mask] - split) / ((zmax - zmin) / 4)
        colors[mask, 2] = 1

        return colors

