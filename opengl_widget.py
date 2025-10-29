# -*- coding: utf-8 -*-
# @Author  : LG
import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QVector3D, QMatrix4x4, QQuaternion, QVector2D, QVector4D, QPolygonF
from PyQt5.QtCore import QPointF
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.GL import *
import numpy as np
from camera import Camera
from transform import Transform
from math import pi, cos, sin
from enum import Enum
from collections import namedtuple
# Removed dependency on imgviz to simplify packaging (avoids pulling matplotlib in analysis)
def label_colormap(n=256):
    """
    Generate a label colormap similar to PASCAL/VOC as uint8 array of shape (n, 3).
    """
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid & 1) << (7 - j))
            g |= (((cid >> 1) & 1) << (7 - j))
            b |= (((cid >> 2) & 1) << (7 - j))
            cid >>= 3
        cmap[i] = [r, g, b]
    return cmap
from configs import MODE, DISPLAY
import datetime
import json

CATEGORY = namedtuple('CATEGORY', ['id', 'name', 'color'])

class GLWidget(QGLWidget):
    def __init__(self, parent, mainwindow):
        super(GLWidget, self).__init__(parent)
        self.setMouseTracking(True)
        self.mainwindow = mainwindow
        self.pointcloud = None
        self.categorys:np.ndarray = None
        self.instances:np.ndarray = None

        self.mask:np.ndarray = None                 # 显示用掩码
        self.category_display_state_dict = {}
        self.instance_display_state_dict = {}

        self.current_vertices:np.ndarray = None       # doubel
        self.current_colors:np.ndarray = None       # doubel
        self.elevation_color:np.ndarray = None
        self.category_color:np.ndarray = None
        self.instance_color:np.ndarray = None


        self.mode:MODE = MODE.VIEW_MODE
        self.display:DISPLAY = DISPLAY.RGB

        self.point_size = 1
        self.pickpoint_radius = 1  # 点拾取半径
        self.show_size:int = 60
        self.show_circle = False
        self.ortho_change_scale = 1
        self.ortho_change_rate = 0.95
        self.center_vertex = QVector3D(0, 0, 0)

        self.vertex_transform = Transform()
        self.circle_transform = Transform()
        self.axes_transform = Transform()
        self.keep_transform = Transform()
        self.projection = QMatrix4x4()
        self.camera = Camera()

        self.color_map = label_colormap(256).astype(np.float32) / 255
        self.color_map[0] = [1, 1, 1]
        self.mouse_left_button_pressed = False
        self.mouse_right_button_pressed = False
        self.lastX, self.lastY = None, None
        self.polygon_vertices:list = []
        self.filter_mode:bool = False
        # Track whether a filtered subset is currently active
        self.is_filtered:bool = False
        # Enable setting rotation center via left double-click
        self.rotation_center_on_double_click_enabled:bool = True

        # Projection overlay elements
        self.projection_enabled: bool = False
        self.proj_K = None                   # 3x3 intrinsics
        self.proj_dist = None                # distortion coefficients [k1,k2,p1,p2,k3] (OpenCV-style)
        self.proj_T_lidar_cam = None         # 4x4 lidar->camera extrinsic
        self.proj_image_folder: str = None   # optional override folder
        self.proj_image_ext: str = '.jpg'    # default extension
        self.proj_max_points: int = 200000   # cap to avoid UI stalls
        # Undistort-and-project pipeline control (default on; fallbacks gracefully if OpenCV not present)
        self.proj_use_undistort: bool = True
        self._proj_warned_no_cv2: bool = False
        # Adjustable overlay sizing and point size
        self.proj_overlay_ratio: float = 0.45  # fraction of GL width (CTRL + mouse wheel to change)
        self.proj_image_aspect: float = 1.5    # width/height (updated from image)
        self.proj_point_size: int = 3          # projected point size in pixels (SHIFT + mouse wheel)

        self.proj_label = QtWidgets.QLabel(self)
        self.proj_label.setVisible(False)
        self.proj_label.setStyleSheet("background-color: rgba(0,0,0,180); border: 1px solid #444;")
        # dynamic size computed in _layout_projection_overlay()
        self.proj_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        # Don't block interactions with GL canvas
        self.proj_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        # GL resource handles to allow proper deletion and avoid leaks
        self.vertex_vao = None
        self.vertex_vbos = None
        self.axes_vbos = None
        self.circle_vbos = None
        self.polygon_vaos = None
        self.polygon_vbos = None

    def init_vertex_shader(self):
        vertex_src = """
        # version 330 core
        layout(location = 0) in vec3 a_pos;
        layout(location = 1) in vec3 a_color;
        out vec3 v_color;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;

        void main(){
            gl_Position = projection * view * model * vec4(a_pos, 1.0);
            v_color = a_color;
        }
        """
        fragment_src = """
        # version 330 core
        in vec3 v_color;
        out vec4 out_color;

        void main(){
            out_color = vec4(v_color, 1.0);
        }
        """
        self.vertex_shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        glUseProgram(self.vertex_shader)

        self.model_loc = glGetUniformLocation(self.vertex_shader, 'model')
        self.view_loc = glGetUniformLocation(self.vertex_shader, 'view')
        self.proj_loc = glGetUniformLocation(self.vertex_shader, 'projection')
        glUseProgram(0)

    def init_vertex_vao(self):
        # Ensure we have a current GL context for deletion/creation
        try:
            self.makeCurrent()
        except Exception:
            pass

        # Delete previous VAO/VBOs to avoid GPU memory leaks
        try:
            if getattr(self, 'vertex_vao', None):
                glDeleteVertexArrays(1, [self.vertex_vao])
        except Exception:
            pass
        try:
            if getattr(self, 'vertex_vbos', None):
                glDeleteBuffers(len(self.vertex_vbos), self.vertex_vbos)
        except Exception:
            pass

        # Create fresh VAO/VBOs
        self.vertex_vao = glGenVertexArrays(1)
        self.vertex_vbos = glGenBuffers(2)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbos[0])
        glBufferData(GL_ARRAY_BUFFER, self.current_vertices.nbytes, self.current_vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbos[1])
        glBufferData(GL_ARRAY_BUFFER, self.current_colors.nbytes, self.current_colors, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(self.vertex_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbos[0])
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.current_vertices.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbos[1])
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.current_colors.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(0)

    def init_axes_vao(self):
        def createcircle(center:QVector3D, radius, num_point):
            xs, ys, zs = [], [], []
            angle = pi * 2 / num_point
            cx, cy, cz = center.x(), center.y(), center.z()
            for i in range(num_point):
                xs.append(cx + radius * cos(angle * i))
                ys.append(cy + radius * sin(angle * i))
                zs.append(cz)
            return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(zs, dtype=np.float32)

        num_point = 360
        radius = 1
        # 轴
        self.axes_x_vertices = np.array([0, 0, 0, radius, 0, 0], dtype=np.float32)
        self.axes_x_colors = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint16)
        self.axes_y_vertices = np.array([0, 0, 0, 0, radius, 0], dtype=np.float32)
        self.axes_y_colors = np.array([0, 1, 0, 0, 1, 0], dtype=np.uint16)
        self.axes_z_vertices = np.array([0, 0, 0, 0, 0, radius], dtype=np.float32)
        self.axes_z_colors = np.array([0, 0, 1, 0, 0, 1], dtype=np.uint16)

        self.axes_vaos = glGenVertexArrays(3)   # xyz轴
        axes_vbos = glGenBuffers(6)
        self.axes_vbos = axes_vbos

        glBindVertexArray(self.axes_vaos[0])  # x轴
        glBindBuffer(GL_ARRAY_BUFFER, axes_vbos[0])
        glBufferData(GL_ARRAY_BUFFER, self.axes_x_vertices.nbytes, self.axes_x_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.axes_x_vertices.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, axes_vbos[1])
        glBufferData(GL_ARRAY_BUFFER, self.axes_x_colors.nbytes, self.axes_x_colors, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_UNSIGNED_SHORT, GL_FALSE, self.axes_x_colors.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glBindVertexArray(self.axes_vaos[1])  # y轴
        glBindBuffer(GL_ARRAY_BUFFER, axes_vbos[2])
        glBufferData(GL_ARRAY_BUFFER, self.axes_y_vertices.nbytes, self.axes_y_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.axes_y_vertices.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, axes_vbos[3])
        glBufferData(GL_ARRAY_BUFFER, self.axes_y_colors.nbytes, self.axes_y_colors, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_UNSIGNED_SHORT, GL_FALSE, self.axes_y_colors.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glBindVertexArray(self.axes_vaos[2])  # z轴
        glBindBuffer(GL_ARRAY_BUFFER, axes_vbos[4])
        glBufferData(GL_ARRAY_BUFFER, self.axes_z_vertices.nbytes, self.axes_z_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.axes_z_vertices.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, axes_vbos[5])
        glBufferData(GL_ARRAY_BUFFER, self.axes_z_colors.nbytes, self.axes_z_colors, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_UNSIGNED_SHORT, GL_FALSE, self.axes_z_colors.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # 环
        xs, ys, zs = createcircle(QVector3D(0, 0, 0), radius, num_point)
        self.circle_x_vertices = np.dstack([zs, xs, ys])
        self.circle_x_colors = np.array([1, 0, 0] * num_point, dtype=np.uint16)
        self.circle_y_vertices = np.dstack([ys, zs, xs])
        self.circle_y_colors = np.array([0, 1, 0] * num_point, dtype=np.uint16)
        self.circle_z_vertices = np.dstack([xs, ys, zs])
        self.circle_z_colors = np.array([0, 0, 1] * num_point, dtype=np.uint16)

        self.circle_vaos = glGenVertexArrays(3)  # xyz环
        circle_vbos = glGenBuffers(6)
        self.circle_vbos = circle_vbos

        glBindVertexArray(self.circle_vaos[0])  # x环
        glBindBuffer(GL_ARRAY_BUFFER, circle_vbos[0])
        glBufferData(GL_ARRAY_BUFFER, self.circle_x_vertices.nbytes, self.circle_x_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.circle_x_vertices.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, circle_vbos[1])
        glBufferData(GL_ARRAY_BUFFER, self.circle_x_colors.nbytes, self.circle_x_colors, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_UNSIGNED_SHORT, GL_FALSE, self.circle_x_colors.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glBindVertexArray(self.circle_vaos[1])  # y环
        glBindBuffer(GL_ARRAY_BUFFER, circle_vbos[2])
        glBufferData(GL_ARRAY_BUFFER, self.circle_y_vertices.nbytes, self.circle_y_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.circle_y_vertices.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, circle_vbos[3])
        glBufferData(GL_ARRAY_BUFFER, self.circle_y_colors.nbytes, self.circle_y_colors, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_UNSIGNED_SHORT, GL_FALSE, self.circle_y_colors.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glBindVertexArray(self.circle_vaos[2])  # z环
        glBindBuffer(GL_ARRAY_BUFFER, circle_vbos[4])
        glBufferData(GL_ARRAY_BUFFER, self.circle_z_vertices.nbytes, self.circle_z_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.circle_z_vertices.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, circle_vbos[5])
        glBufferData(GL_ARRAY_BUFFER, self.circle_z_colors.nbytes, self.circle_z_colors, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_UNSIGNED_SHORT, GL_FALSE, self.circle_z_colors.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def init_polygon_vao(self):
        if self.polygon_vertices is None:
            return

        # Ensure current context; delete previous polygon buffers to avoid leaks
        try:
            self.makeCurrent()
        except Exception:
            pass
        try:
            if getattr(self, 'polygon_vaos', None):
                glDeleteVertexArrays(1, [self.polygon_vaos])
        except Exception:
            pass
        try:
            if getattr(self, 'polygon_vbos', None):
                glDeleteBuffers(len(self.polygon_vbos), self.polygon_vbos)
        except Exception:
            pass

        polygon_vertices = np.array(self.polygon_vertices, dtype=np.float32)
        self.polygon_colors = np.array([0.5, 0.5, 0.5] * polygon_vertices.shape[0], dtype=np.uint16)

        self.polygon_vaos = glGenVertexArrays(1)
        self.polygon_vbos = glGenBuffers(2)

        glBindVertexArray(self.polygon_vaos)
        glBindBuffer(GL_ARRAY_BUFFER, self.polygon_vbos[0])
        glBufferData(GL_ARRAY_BUFFER, polygon_vertices.nbytes, polygon_vertices, GL_STREAM_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, polygon_vertices.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, self.polygon_vbos[1])
        glBufferData(GL_ARRAY_BUFFER, self.polygon_colors.nbytes, self.polygon_colors, GL_STREAM_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_UNSIGNED_SHORT, GL_FALSE, self.polygon_colors.itemsize * 3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def initializeGL(self):
        self.init_vertex_shader()
        self.init_axes_vao()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.44, 0.62, 0.81, 1)

    def set_background_color(self, red, green, blue, alpha=1):
        glClearColor(red, green, blue, alpha)
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.paintAxes()
        self.paintCircle()
        glPointSize(self.point_size)

        if self.pointcloud is not None:
            glUseProgram(self.vertex_shader)
            glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.vertex_transform.toMatrix().data())
            glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.camera.toMatrix().data())
            glBindVertexArray(self.vertex_vao)
            glDrawArrays(GL_POINTS, 0, int(self.current_vertices.nbytes / self.current_vertices.itemsize/3))
            glUseProgram(0)

        if self.polygon_vertices is not None:
            glUseProgram(self.vertex_shader)
            glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.keep_transform.toMatrix().data())
            glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.camera.toMatrix().data())

            self.init_polygon_vao()
            glBindVertexArray(self.polygon_vaos)
            polygon_vertices = np.array(self.polygon_vertices, dtype=np.float32)
            glDrawArrays(GL_LINE_STRIP, 0, int(polygon_vertices.nbytes / polygon_vertices.itemsize / 3))
            glUseProgram(0)

    def paintAxes(self):
        glUseProgram(self.vertex_shader)
        # 坐标固定位置
        self.axes_transform.setTranslation((self.width() / 2 - self.show_size - 20) * self.ortho_change_scale,
                                           (self.height() / 2 - self.show_size - 20) * self.ortho_change_scale,
                                           1000)
        self.axes_transform.setScale(self.show_size * self.ortho_change_scale,
                                     self.show_size * self.ortho_change_scale,
                                     self.show_size * self.ortho_change_scale)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.axes_transform.toMatrix().data())
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.camera.toMatrix().data())
        glBindVertexArray(self.axes_vaos[0])  # x轴
        glDrawArrays(GL_LINE_STRIP, 0, int(self.axes_x_vertices.nbytes / self.axes_x_vertices.itemsize / 3))
        glBindVertexArray(self.axes_vaos[1])  # y轴
        glDrawArrays(GL_LINE_STRIP, 0, int(self.axes_y_vertices.nbytes / self.axes_y_vertices.itemsize / 3))
        glBindVertexArray(self.axes_vaos[2])  # z轴
        glDrawArrays(GL_LINE_STRIP, 0, int(self.axes_z_vertices.nbytes / self.axes_z_vertices.itemsize / 3))
        glUseProgram(0)

    def paintCircle(self):
        glUseProgram(self.vertex_shader)
        # 环固定大小
        scale = (self.height() / 2 * self.ortho_change_scale) / 5 * 4
        self.circle_transform.setScale(scale, scale, scale)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.circle_transform.toMatrix().data())
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.camera.toMatrix().data())
        if self.show_circle == True:
            glBindVertexArray(self.axes_vaos[0])  # x轴
            glDrawArrays(GL_LINE_STRIP, 0, int(self.axes_x_vertices.nbytes / self.axes_x_vertices.itemsize / 3))
            glBindVertexArray(self.axes_vaos[1])  # y轴
            glDrawArrays(GL_LINE_STRIP, 0, int(self.axes_y_vertices.nbytes / self.axes_y_vertices.itemsize / 3))
            glBindVertexArray(self.axes_vaos[2])  # z轴
            glDrawArrays(GL_LINE_STRIP, 0, int(self.axes_z_vertices.nbytes / self.axes_z_vertices.itemsize / 3))
            glBindVertexArray(self.circle_vaos[0])  # x环
            glDrawArrays(GL_LINE_LOOP, 0, int(self.circle_x_vertices.nbytes / self.circle_x_vertices.itemsize / 3))
            glBindVertexArray(self.circle_vaos[1])  # y环
            glDrawArrays(GL_LINE_LOOP, 0, int(self.circle_y_vertices.nbytes / self.circle_y_vertices.itemsize / 3))
            glBindVertexArray(self.circle_vaos[2])  # z环
            glDrawArrays(GL_LINE_LOOP, 0, int(self.circle_z_vertices.nbytes / self.circle_z_vertices.itemsize / 3))
        glUseProgram(0)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        self.projection = QMatrix4x4()
        self.projection.setToIdentity()
        self.projection.ortho(-width / 2 * self.ortho_change_scale, width / 2 * self.ortho_change_scale,
                              -height / 2 * self.ortho_change_scale, height / 2 * self.ortho_change_scale,
                              -300000, 300000)
        glUseProgram(self.vertex_shader)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection.data())
        glUseProgram(0)

        # Layout the bottom-left projection overlay label
        try:
            self._layout_projection_overlay()
        except Exception:
            pass

    def pointsize_change(self, value:int):
        self.point_size += value
        if self.point_size < 1: self.point_size = 1
        if self.point_size > 10: self.point_size = 10
        self.update()

    def reset(self):
        # Free per-cloud GPU buffers (VAO/VBO) to prevent accumulation across loads/filters
        try:
            self.makeCurrent()
        except Exception:
            pass
        # Delete vertex VAO/VBOs
        try:
            if getattr(self, 'vertex_vao', None):
                glDeleteVertexArrays(1, [self.vertex_vao])
        except Exception:
            pass
        try:
            if getattr(self, 'vertex_vbos', None):
                glDeleteBuffers(len(self.vertex_vbos), self.vertex_vbos)
        except Exception:
            pass
        self.vertex_vao = None
        self.vertex_vbos = None
        # Delete polygon VAO/VBOs
        try:
            if getattr(self, 'polygon_vaos', None):
                glDeleteVertexArrays(1, [self.polygon_vaos])
        except Exception:
            pass
        try:
            if getattr(self, 'polygon_vbos', None):
                glDeleteBuffers(len(self.polygon_vbos), self.polygon_vbos)
        except Exception:
            pass
        self.polygon_vaos = None
        self.polygon_vbos = None

        # Reset data references to release CPU memory
        self.pointcloud = None
        self.mask:np.ndarray = None
        self.current_vertices: np.ndarray = None
        self.current_colors:np.ndarray = None
        self.elevation_color:np.ndarray = None
        self.instance_color:np.ndarray = None
        self.category_color:np.ndarray = None
        self.categorys:np.ndarray = None
        self.instances:np.ndarray = None

        # Reset transforms and viewport scale
        self.vertex_transform.__init__()
        self.circle_transform.__init__()
        self.camera.__init__()
        self.axes_transform.__init__()
        self.ortho_change_scale = 1

        self.resizeGL(self.width(), self.height())

    def load_vertices(self, pointcloud, categorys:np.ndarray=None, instances:np.ndarray=None):
        self.reset()

        self.pointcloud = pointcloud
        self.vertex_transform.setTranslation(-pointcloud.size[0]/2, -pointcloud.size[1]/2, -pointcloud.size[2]/2)
        self.mask = np.ones(pointcloud.num_point, dtype=bool)
        self.category_display_state_dict = {}
        self.instance_display_state_dict = {}

        self.categorys = categorys if categorys is not None else np.zeros(pointcloud.num_point, dtype=np.int16)
        self.instances = instances if instances is not None else np.zeros(pointcloud.num_point, dtype=np.int16)
        self.ortho_change_scale = max(pointcloud.size[0] / (self.height() / 5 * 4),
                                      pointcloud.size[1] / (self.height() / 5 * 4))
        self.current_vertices = self.pointcloud.xyz
        self.current_colors = self.pointcloud.rgb
        self.init_vertex_vao()
        self.resizeGL(self.width(), self.height())
        self.update()

    def change_mode_to_pick(self):
        # Enter polygon annotation mode (never filter on right-click here)
        self.filter_mode = False
        if self.mode == MODE.VIEW_MODE:
            self.mode = MODE.DRAW_MODE
        self.polygon_vertices = []

    def change_mode_to_view(self):
        if self.mode == MODE.DRAW_MODE:
            self.mode = MODE.VIEW_MODE
            self.polygon_vertices = []

    def toggle_filter_mode(self):
        """
        Toggle polygon filtering workflow:
        - If currently drawing a filter polygon, clicking Filter will cancel and restore all points.
        - If a filtered subset is active, clicking Filter will restore all points.
        - Otherwise, clicking Filter enters filter drawing mode.
        """
        if self.mode == MODE.DRAW_MODE and self.filter_mode:
            # Currently in filter drawing mode -> cancel and restore all points
            self.filter_mode = False
            self.mode = MODE.VIEW_MODE
            self.polygon_vertices = []
            if self.mask is not None:
                self.mask.fill(True)
            self.is_filtered = False
            self.category_display_state_dict = {}
            self.instance_display_state_dict = {}
            # Refresh display according to current mode
            if self.display == DISPLAY.ELEVATION:
                if self.elevation_color is None:
                    self.parent().elevation_color_thread_start()
                else:
                    self.change_color_to_elevation()
            elif self.display == DISPLAY.RGB:
                self.change_color_to_rgb()
            elif self.display == DISPLAY.CATEGORY:
                self.change_color_to_category()
            elif self.display == DISPLAY.INSTANCE:
                self.change_color_to_instance()
        else:
            # Not in filter draw mode
            if getattr(self, 'is_filtered', False):
                # A filtered subset is active -> restore all points
                self.is_filtered = False
                if self.mask is not None:
                    self.mask.fill(True)
                self.category_display_state_dict = {}
                self.instance_display_state_dict = {}
                if self.display == DISPLAY.ELEVATION:
                    if self.elevation_color is None:
                        self.parent().elevation_color_thread_start()
                    else:
                        self.change_color_to_elevation()
                elif self.display == DISPLAY.RGB:
                    self.change_color_to_rgb()
                elif self.display == DISPLAY.CATEGORY:
                    self.change_color_to_category()
                elif self.display == DISPLAY.INSTANCE:
                    self.change_color_to_instance()
            else:
                # Enter filter drawing mode
                self.filter_mode = True
                self.polygon_vertices = []
                self.mode = MODE.DRAW_MODE

    def change_color_to_rgb(self):
        if self.pointcloud is None:
            return
        self.current_vertices = self.pointcloud.xyz[self.mask]
        self.current_colors = self.pointcloud.rgb[self.mask]
        self.display = DISPLAY.RGB
        self.init_vertex_vao()
        self.update()
        self.mainwindow.update_dock()
        # If projection overlay is enabled, refresh it
        self._maybe_refresh_projection()

    def change_color_to_category(self):
        if self.pointcloud is None:
            return
        self.category_color_update()
        self.current_vertices = self.pointcloud.xyz[self.mask]
        self.current_colors = self.category_color[self.mask]
        self.display = DISPLAY.CATEGORY

        self.init_vertex_vao()
        self.update()
        self.mainwindow.update_dock()
        # If projection overlay is enabled, refresh it (colors may have changed)
        self._maybe_refresh_projection()

    def change_color_to_instance(self):
        if self.pointcloud is None:
            return
        # if self.instance_color is None:
        self.instance_color_update()
        self.current_vertices = self.pointcloud.xyz[self.mask]
        self.current_colors = self.instance_color[self.mask]
        self.display = DISPLAY.INSTANCE
        self.init_vertex_vao()
        self.update()
        self.mainwindow.update_dock()
        # Projection overlay uses category colors; still refresh to reflect mask changes
        self._maybe_refresh_projection()

    def change_color_to_elevation(self):
        if self.pointcloud is None:
            return
        if self.elevation_color is None:
            self.parent().elevation_color_thread_start()
            return
        self.current_vertices = self.pointcloud.xyz[self.mask]
        self.current_colors = self.elevation_color[self.mask]
        self.display = DISPLAY.ELEVATION
        self.init_vertex_vao()
        self.update()
        self.mainwindow.update_dock()
        # Projection overlay does not show elevation, but mask may have changed
        self._maybe_refresh_projection()

    def category_color_update(self):
        if self.categorys is None:
            return
        self.category_color = np.zeros(self.pointcloud.xyz.shape, dtype=np.float32)
        for id, (category, color) in enumerate(self.parent().category_color_dict.items()):
            color = QtGui.QColor(color)
            self.category_color[self.categorys==id] = (color.redF(), color.greenF(), color.blueF())

    def instance_color_update(self):
        if self.instances is None:
            return
        self.instance_color = np.zeros(self.pointcloud.xyz.shape, dtype=np.float32)
        self.instance_color = self.color_map[self.instances]

    def reapply_category_colors(self):
        """
        Clamp/validate category indices to current config and refresh category coloring.
        """
        if self.pointcloud is None or self.categorys is None:
            return
        try:
            categories = list(self.parent().category_color_dict.keys())
            n = len(categories)
            if n == 0:
                return
            uncls_index = categories.index('__unclassified__') if '__unclassified__' in categories else 0

            cats = self.categorys.astype(np.int32, copy=False)
            valid_mask = (cats >= 0) & (cats < n)
            if not np.all(valid_mask):
                cats = np.where(valid_mask, cats, uncls_index).astype(np.int16)
                self.categorys = cats

            # Recompute colors and switch to category view
            self.category_color_update()
            self.change_color_to_category()
            if hasattr(self, 'mainwindow') and self.mainwindow is not None:
                self.mainwindow.save_state = False
                self.mainwindow.show_message("Reapplied category colors.", 3000)
        except Exception as e:
            if hasattr(self, 'mainwindow') and self.mainwindow is not None:
                self.mainwindow.show_message(f"Reapply failed: {e}", 5000)
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.lastX, self.lastY = event.pos().x(), event.pos().y()

        if self.mode == MODE.DRAW_MODE:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                x, y = event.pos().x() - self.width() / 2, self.height() / 2 - event.pos().y()
                x = x * self.ortho_change_scale
                y = y * self.ortho_change_scale

                if not self.polygon_vertices:
                    self.polygon_vertices = [[x, y, 10000], [x, y, 10000], [x, y, 10000]]
                else:
                    self.polygon_vertices.insert(-1, [x, y, 10000])
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                if self.filter_mode:
                    self.apply_polygon_filter()
                else:
                    # 选择类别与group
                    self.mainwindow.category_choice_dialog.load_cfg()
                    self.mainwindow.category_choice_dialog.show()
        else:
            self.show_circle = True
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.mouse_left_button_pressed = True
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self.mouse_right_button_pressed = True
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self.show_circle = False
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.mouse_left_button_pressed = False
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self.mouse_right_button_pressed = False
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        xpos, ypos = event.pos().x(), event.pos().y()

        if self.mode == MODE.VIEW_MODE:
            if self.mouse_left_button_pressed or self.mouse_right_button_pressed:
                xoffset = xpos - self.lastX
                yoffset = ypos - self.lastY
                self.lastX = xpos
                self.lastY = ypos
                if self.mouse_left_button_pressed:
                    self.mouse_rotate(xoffset, yoffset)
                elif self.mouse_right_button_pressed:
                    self.mouse_move(xoffset, yoffset)
                else:
                    pass

        elif self.mode == MODE.DRAW_MODE:
            x, y = event.pos().x() - self.width() / 2, self.height() / 2 - event.pos().y()
            x = x * self.ortho_change_scale
            y = y * self.ortho_change_scale

            if len(self.polygon_vertices) > 2 :
                self.polygon_vertices[-2] = [x, y, 10000]   # 更新当前点

        self.update()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        # Left-button double-click: set nearest on-screen point as rotation center
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.pointcloud is None:
                return
            if getattr(self, "rotation_center_on_double_click_enabled", False):
                self.set_rotation_center_nearest_screen_point(event)
                self.update()
                return
            # Fallback to previous ray-based pick when the feature is disabled
            x, y = event.pos().x(), self.height() - event.pos().y()
            point = self.pickpoint(x, y)
            if point.size:
                self.vertex_transform.setTranslationwithRotate(-point[0], -point[1], -point[2])
            self.update()

    def cache_pick(self):
        self.change_mode_to_view()
        # Ensure filter mode is off when caching/clearing selection
        self.filter_mode = False
        self.mask.fill(True)
        self.category_display_state_dict = {}
        self.instance_display_state_dict = {}

        self.mainwindow.save_state = False
        if self.display == DISPLAY.ELEVATION:
            self.change_color_to_elevation()
        elif self.display == DISPLAY.RGB:
            self.change_color_to_rgb()
        elif self.display == DISPLAY.CATEGORY:
            self.change_color_to_category()
        elif self.display == DISPLAY.INSTANCE:
            self.change_color_to_instance()

    def apply_polygon_filter(self):
        # Apply polygon selection to filter displayed points
        if self.pointcloud is None or not self.polygon_vertices:
            self.change_mode_to_view()
            return

        # Convert polygon to screen coordinates
        polygon_vertices = [p[:] for p in self.polygon_vertices]
        for p in polygon_vertices:
            p[0] = p[0] / self.ortho_change_scale + self.width() / 2
            p[1] = self.height() / 2 - p[1] / self.ortho_change_scale
        polygon = QPolygonF([QPointF(p[0], p[1]) for p in polygon_vertices])
        rect = polygon.boundingRect()

        # Project all vertices (unmasked) to 2D screen space
        projection = np.array(self.projection.data()).reshape(4, 4)
        camera = np.array(self.camera.toMatrix().data()).reshape(4, 4)
        vertex_transform = np.array(self.vertex_transform.toMatrix().data()).reshape(4, 4)
        verts = np.hstack((self.pointcloud.xyz, np.ones((self.pointcloud.xyz.shape[0], 1), dtype=np.float32)))
        verts2model = verts.dot(vertex_transform.dot(camera))
        verts2projection = verts2model.dot(projection)
        xys = verts2projection[:, :2]
        xys = xys + np.array((1, -1))
        xys = xys * np.array((self.width() / 2, self.height() / 2)) + 1.0
        xys = xys * np.array((1, -1))

        l, r, t, b = rect.x(), rect.x() + rect.width(), rect.y(), rect.y() + rect.height()
        mask1 = (l < xys[:, 0]) & (xys[:, 0] < r) & (t < xys[:, 1]) & (xys[:, 1] < b)
        mask2 = [polygon.containsPoint(QPointF(p[0], p[1]), QtCore.Qt.FillRule.WindingFill) for p in xys[mask1]]
        mask1[mask1 == True] = mask2

        # Set mask to selected points only
        self.mask = mask1

        # Exit draw mode
        self.change_mode_to_view()
        # Filter selection applied; mark filtered state active
        self.is_filtered = True
        self.filter_mode = False
        self.polygon_vertices = []

        # Refresh display to reflect new mask
        if self.display == DISPLAY.ELEVATION:
            if self.elevation_color is None:
                self.parent().elevation_color_thread_start()
            else:
                self.change_color_to_elevation()
        elif self.display == DISPLAY.RGB:
            self.change_color_to_rgb()
        elif self.display == DISPLAY.CATEGORY:
            self.change_color_to_category()
        elif self.display == DISPLAY.INSTANCE:
            self.change_color_to_instance()

    def polygon_pick(self, category:int=None, instance:int=None):
        polygon_vertices = self.polygon_vertices
        """
        x, y = event.pos().x() - self.width() / 2, self.height() / 2 - event.pos().y()
                x = x * self.ortho_change_scale
                y = y * self.ortho_change_scale
        """
        for p in polygon_vertices:
            p[0] = p[0] / self.ortho_change_scale + self.width() / 2
            p[1] = self.height() / 2 - p[1] / self.ortho_change_scale
        polygon_vertices = [QPointF(p[0], p[1]) for p in polygon_vertices]
        polygon = QPolygonF(polygon_vertices)
        rect = polygon.boundingRect()

        vertices2D = self.vertices_to_2D()
        l, r, t, b = rect.x(), rect.x() + rect.width(), rect.y(), rect.y() + rect.height()

        mask1 = (l < vertices2D[:, 0]) & (vertices2D[:, 0] < r) & \
               (t < vertices2D[:, 1]) & (vertices2D[:, 1] < b)
        print('mask1: ', sum(mask1))
        mask2 = [polygon.containsPoint(QPointF(p[0], p[1]), QtCore.Qt.FillRule.WindingFill) for p in vertices2D[mask1]]

        print('mask2: ', sum(mask2))
        mask1[mask1 == True] = mask2
        mask = self.mask.copy()
        mask[mask==True] = mask1
        if instance is not None:
            self.instances[mask] = instance
        if category is not None:
            index = list(self.mainwindow.category_color_dict.keys()).index(category)
            self.categorys[mask] = index

        #
        # Keep current filtered mask; do not restore all points here
        self.category_display_state_dict = {}
        self.instance_display_state_dict = {}

        self.mainwindow.save_state = False
        if self.display == DISPLAY.ELEVATION:
            self.change_color_to_category()
        elif self.display == DISPLAY.RGB:
            self.change_color_to_category()
        elif self.display == DISPLAY.CATEGORY:
            self.change_color_to_category()
        elif self.display == DISPLAY.INSTANCE:
            self.change_color_to_category()

    def pickpoint(self, x, y):
        if self.pointcloud is None:return np.array([])

        point1 = QVector3D(x, y, 0).unproject(self.camera.toMatrix() * self.vertex_transform.toMatrix(),
                                              self.projection,
                                              QtCore.QRect(0, 0, self.width(), self.height()))
        point2 = QVector3D(x, y, 1).unproject(self.camera.toMatrix() * self.vertex_transform.toMatrix(),
                                              self.projection,
                                              QtCore.QRect(0, 0, self.width(), self.height()))
        vector = (point2 - point1)  # 直线向量
        vector.normalize()

        # 点到直线（点向式）的距离
        t = (vector.x() * (self.current_vertices[:, 0] - point1.x()) +
             vector.y() * (self.current_vertices[:, 1] - point1.y()) +
             vector.z() * (self.current_vertices[:, 2] - point1.z())) / (vector.x() ** 2 + vector.y() ** 2 + vector.z() ** 2)

        d = (self.current_vertices[:, 0] - (vector.x() * t + point1.x())) ** 2 + \
            (self.current_vertices[:, 1] - (vector.y() * t + point1.y())) ** 2 + \
            (self.current_vertices[:, 2] - (vector.z() * t + point1.z())) ** 2

        pickpoint_radius = self.pickpoint_radius * self.ortho_change_scale
        mask = d < pickpoint_radius**2

        if not any(mask):
            return np.array([])

        mask1 = self.mask.copy()
        mask1[mask1==True] = mask
        points = self.pointcloud.xyz[mask1]
        index = np.argmin(points[:, 0] * vector.x() + points[:, 1] * vector.y() + points[:, 2] * vector.z())
        point = points[index]   # 取最近的点
        return point

    def set_rotation_center_nearest_screen_point(self, event: QtGui.QMouseEvent):
        """
        Set rotation center to the nearest on-screen point to the mouse cursor.
        Uses 2D screen-space distance among currently displayed (masked) points.
        """
        x = event.pos().x()
        y = event.pos().y()
        vertices2D = self.vertices_to_2D()
        if vertices2D is None or vertices2D.shape[0] == 0:
            return

        # Compute nearest 2D point index
        dx = vertices2D[:, 0] - x
        dy = vertices2D[:, 1] - y
        idx = int(np.argmin(dx * dx + dy * dy))
        point = self.current_vertices[idx]

        # Store and apply rotation center by translating such that the point is at origin
        self.center_vertex = QVector3D(point[0], point[1], point[2])
        self.vertex_transform.setTranslationwithRotate(-point[0], -point[1], -point[2])

    def set_rotation_center_on_double_click_enabled(self, enabled: bool):
        self.rotation_center_on_double_click_enabled = enabled
        if self.mainwindow:
            self.mainwindow.show_message("Rotation center on double-click: {}".format("ON" if enabled else "OFF"), 2000)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        # CTRL + Wheel: resize projection overlay; SHIFT + Wheel: change projected point size
        mods = event.modifiers()
        try:
            if mods & QtCore.Qt.ControlModifier:
                delta = event.angleDelta().y()
                step = 0.05
                if delta > 0:
                    self.proj_overlay_ratio = min(0.95, self.proj_overlay_ratio + step)
                elif delta < 0:
                    self.proj_overlay_ratio = max(0.20, self.proj_overlay_ratio - step)
                self._layout_projection_overlay()
                if hasattr(self, 'mainwindow') and self.mainwindow:
                    self.mainwindow.show_message(f"Overlay size: {int(self.proj_overlay_ratio * 100)}%", 1500)
                self._maybe_refresh_projection()
                return
            elif mods & QtCore.Qt.ShiftModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.proj_point_size = min(10, self.proj_point_size + 1)
                elif delta < 0:
                    self.proj_point_size = max(1, self.proj_point_size - 1)
                if hasattr(self, 'mainwindow') and self.mainwindow:
                    self.mainwindow.show_message(f"Projection point size: {self.proj_point_size}px", 1500)
                self._maybe_refresh_projection()
                return
        except Exception:
            pass
        # Default: zoom viewport
        self.ortho_area_change(event)
        self.update()

    def mouse_rotate(self, xoffset, yoffset):
        # 点云旋转
        self.vertex_transform.rotate(self.vertex_transform.localup, xoffset * 0.5)
        self.vertex_transform.rotate(self.vertex_transform.localright, yoffset * 0.5)
        # 坐标旋转
        self.circle_transform.rotate_in_place(self.circle_transform.localup, xoffset * 0.5)
        self.circle_transform.rotate_in_place(self.circle_transform.localright, yoffset * 0.5)
        self.axes_transform.rotate_in_place(self.axes_transform.localup, xoffset * 0.5)
        self.axes_transform.rotate_in_place(self.axes_transform.localright, yoffset * 0.5)
        self.update()

    def mouse_move(self, xoffset, yoffset):
        self.vertex_transform.translate(xoffset * self.ortho_change_scale, -yoffset * self.ortho_change_scale, 0)
        self.update()

    def ortho_area_change(self, event: QtGui.QWheelEvent):
        angle = event.angleDelta().y()
        if angle < 0:
            self.ortho_change_scale /= self.ortho_change_rate
        elif angle > 0:
            self.ortho_change_scale *= self.ortho_change_rate
        else:
            return
        self.resizeGL(self.width(), self.height())

    def set_right_view(self):
        self.vertex_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.circle_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.axes_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))

        self.vertex_transform.rotate(QVector3D(0, 1, 0), -90)
        self.circle_transform.rotate(QVector3D(0, 1, 0), -90)
        self.axes_transform.rotate(QVector3D(0, 1, 0), -90)
        self.update()

    def set_back_view(self):
        self.vertex_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.circle_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.axes_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.vertex_transform.rotate(QVector3D(0, 1, 0), 180)
        self.circle_transform.rotate(QVector3D(0, 1, 0), 180)
        self.axes_transform.rotate(QVector3D(0, 1, 0), 180)
        self.update()

    def set_top_view(self):
        self.vertex_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), 0))
        self.circle_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), 0))
        self.axes_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 1), 0))
        self.update()

    def set_left_view(self):
        self.vertex_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.circle_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.axes_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.vertex_transform.rotate(QVector3D(0, 1, 0), 90)
        self.circle_transform.rotate(QVector3D(0, 1, 0), 90)
        self.axes_transform.rotate(QVector3D(0, 1, 0), 90)
        self.update()

    def set_front_view(self):
        self.vertex_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.circle_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.axes_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 270))
        self.update()

    def set_bottom_view(self):
        self.vertex_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(0, 1, 0), -180))
        self.circle_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(0, 1, 0), -180))
        self.axes_transform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(0, 1, 0), -180))
        self.update()

    def vertices_to_2D(self):
        if self.pointcloud is None:
            return
        # 转numpy便于计算
        projection = np.array(self.projection.data()).reshape(4, 4)
        camera = np.array(self.camera.toMatrix().data()).reshape(4, 4)
        vertex_transform = np.array(self.vertex_transform.toMatrix().data()).reshape(4, 4)
        # 添加维度
        vertexs = np.hstack((self.current_vertices, np.ones(shape=(self.current_vertices.shape[0], 1))))
        vertexs2model = vertexs.dot(vertex_transform.dot(camera))
        vertexs2projection = vertexs2model.dot(projection)
        # 转换到屏幕坐标
        xys = vertexs2projection[:, :2]
        xys = xys + np.array((1, -1))
        xys = xys * np.array((self.width() / 2, self.height() / 2)) + 1.0
        xys = xys * np.array((1, -1))
        return xys

    def save_windows_to_pic(self):
        pix = glReadPixels(0, 0, self.width(), self.height(), GL_RGB, GL_UNSIGNED_BYTE)
        img = QtGui.QImage(pix, self.width(), self.height(), QtGui.QImage.Format_RGB888)
        img = img.mirrored(False, True)

        image_name = "{}.jpg".format(datetime.datetime.utcnow())
        save_path = os.path.join(self.mainwindow.current_root, image_name)
        img.save(save_path)
        print("save image to ", save_path)

    def _maybe_refresh_projection(self):
        """Refresh bottom-left projection overlay if enabled and a calibration is loaded."""
        try:
            if getattr(self, "projection_enabled", False) and self.proj_K is not None and self.proj_T_lidar_cam is not None:
                cur = getattr(self.mainwindow, "current_file", None)
                if isinstance(cur, str) and len(cur) > 0:
                    self.update_projection_for_current(cur)
        except Exception:
            pass

    # ===== Projection overlay (bottom-left) =====

    def _layout_projection_overlay(self):
        """Place and size the overlay label at bottom-left with small margin."""
        if getattr(self, "proj_label", None) is None:
            return
        margin = 10
        # Compute target size based on GL width and image aspect
        target_w = max(50, int(self.width() * float(getattr(self, 'proj_overlay_ratio', 0.45))))
        aspect = float(getattr(self, 'proj_image_aspect', 1.5))
        target_h = max(50, int(target_w / aspect)) if aspect > 1e-6 else 200
        # Prevent overlay from exceeding 70% of GL height
        max_h = int(self.height() * 0.7)
        if target_h > max_h:
            target_h = max_h
            target_w = int(target_h * aspect)
        self.proj_label.setFixedSize(target_w, target_h)
        self.proj_label.move(margin, self.height() - target_h - margin)

    def set_projection_enabled(self, enabled: bool):
        """Toggle overlay visibility; refresh if enabled."""
        self.projection_enabled = bool(enabled)
        if self.proj_label:
            self.proj_label.setVisible(self.projection_enabled)
        # Refresh current cloud's projection when enabling
        try:
            if self.projection_enabled:
                cur = getattr(self.mainwindow, "current_file", None)
                if cur:
                    self.update_projection_for_current(cur)
        except Exception:
            pass

    def load_projection_from_file(self, json_path: str) -> bool:
        """
        Parse calibration.json:
        { "camera_to_image": [[...],[...],[...]],
          "dist": [k1,k2,p1,p2,k3],             // OpenCV style
          "lidar_to_camera": [[...],[...],[...],[...]],
          "image_folder": "optional path",
          "image_ext": ".jpg" }
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            K = np.array(data.get("camera_to_image", []), dtype=np.float64)
            D = np.array(data.get("dist", []), dtype=np.float64).reshape(-1)
            T = np.array(data.get("lidar_to_camera", []), dtype=np.float64)
            if K.shape != (3, 3) or T.shape != (4, 4):
                return False
            self.proj_K = K
            self.proj_dist = D if D.size >= 4 else None
            self.proj_T_lidar_cam = T
            self.proj_image_folder = data.get("image_folder", None)
            ext = data.get("image_ext", None)
            if isinstance(ext, str) and len(ext) > 0:
                self.proj_image_ext = ext if ext.startswith('.') else f'.{ext}'
            return True
        except Exception:
            return False

    def update_projection_for_current(self, pointcloud_path: str):
        """
        Compute and show projection for current cloud using loaded calibration.
        Image filename is derived by replacing extension with self.proj_image_ext.
        Falls back to searching common extensions if not found.
        """
        if not self.projection_enabled:
            return
        if self.pointcloud is None:
            return
        if self.proj_K is None or self.proj_T_lidar_cam is None:
            if self.mainwindow:
                self.mainwindow.show_message("Projection: calibration not loaded.", 4000)
            return

        # Resolve image path
        base = os.path.splitext(os.path.basename(pointcloud_path))[0]
        folder = self.proj_image_folder if self.proj_image_folder else os.path.dirname(pointcloud_path)
        candidates = [os.path.join(folder, base + (self.proj_image_ext or '.jpg'))]
        # Fallbacks
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            p = os.path.join(folder, base + ext)
            if p not in candidates:
                candidates.append(p)
        image_path = None
        for p in candidates:
            if os.path.isfile(p):
                image_path = p
                break
        if image_path is None:
            if self.mainwindow:
                self.mainwindow.show_message("Projection: image file not found.", 4000)
            return

        # Update overlay aspect and layout from image dimensions
        tmp_img = QtGui.QImage(image_path)
        if not tmp_img.isNull() and tmp_img.height() > 0:
            self.proj_image_aspect = float(tmp_img.width()) / float(tmp_img.height())
        self._layout_projection_overlay()

        # Render
        img = self._draw_projection_on_image(image_path)
        if img is None:
            return
        # Fit into label while keeping aspect ratio
        scaled = QtGui.QPixmap.fromImage(
            img.scaled(self.proj_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )
        self.proj_label.setPixmap(scaled)
        self.proj_label.setToolTip(os.path.basename(image_path))

    def _draw_projection_on_image(self, image_path: str) -> QtGui.QImage:
        """Return QImage with all (optionally decimated) points projected and colored by category."""
        # Optionally undistort the image using OpenCV, and use the undistorted intrinsics
        used_K = self.proj_K
        qimg = None
        try:
            if self.proj_use_undistort and (self.proj_dist is not None and self.proj_dist.size >= 4):
                import cv2
                img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if img_bgr is not None and self.proj_K is not None:
                    h, w = img_bgr.shape[:2]
                    K = self.proj_K.astype(np.float64)
                    D = self.proj_dist.astype(np.float64).reshape(-1, 1) if self.proj_dist.ndim == 1 else self.proj_dist.astype(np.float64)
                    K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
                    img_ud = cv2.undistort(img_bgr, K, D, None, K_new)
                    used_K = K_new
                    img_rgb = cv2.cvtColor(img_ud, cv2.COLOR_BGR2RGB)
                    qimg = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.shape[1] * 3, QtGui.QImage.Format_RGB888).copy()
        except Exception as e:
            if not getattr(self, "_proj_warned_no_cv2", False):
                if self.mainwindow:
                    self.mainwindow.show_message(f"Projection: OpenCV undistort unavailable ({e}); using original image.", 4000)
                self._proj_warned_no_cv2 = True
            qimg = None
        if qimg is None:
            qimg = QtGui.QImage(image_path)
            used_K = self.proj_K
        if qimg.isNull():
            if self.mainwindow:
                self.mainwindow.show_message("Projection: failed to load image.", 4000)
            return None
        img_w, img_h = qimg.width(), qimg.height()

        # Prepare 3D points in LiDAR/world coordinates (undo visualization offset)
        xyz = self.pointcloud.xyz
        offset = self.pointcloud.offset if getattr(self.pointcloud, 'offset', None) is not None else np.zeros(3, dtype=np.float32)
        pts = xyz + offset  # shape (N,3)

        # Homogeneous
        N = pts.shape[0]
        ones = np.ones((N, 1), dtype=np.float64)
        pts_h = np.hstack([pts.astype(np.float64), ones])  # (N,4)

        # Transform to camera
        T = self.proj_T_lidar_cam  # (4,4)
        cam = (pts_h @ T.T)  # (N,4)
        X = cam[:, 0]
        Y = cam[:, 1]
        Z = cam[:, 2]

        # Only points in front of camera
        front = Z > 1e-6
        if not np.any(front):
            return qimg
        X = X[front]; Y = Y[front]; Z = Z[front]

        # Normalize
        x = X / Z
        y = Y / Z

        x_d, y_d = x, y

        # Intrinsics (use undistorted intrinsics if image was undistorted)
        fx = used_K[0, 0]
        fy = used_K[1, 1]
        cx = used_K[0, 2]
        cy = used_K[1, 2]
        u = fx * x_d + cx
        v = fy * y_d + cy

        # In-image mask
        in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        if not np.any(in_img):
            return qimg
        u = u[in_img].astype(np.int32)
        v = v[in_img].astype(np.int32)

        # Colors by category
        # Ensure category colors exist
        try:
            if self.category_color is None and self.categorys is not None:
                self.category_color_update()
            colors = (self.category_color[front][in_img] * 255.0).astype(np.uint8) if self.category_color is not None else np.full((u.shape[0], 3), 255, dtype=np.uint8)
        except Exception:
            colors = np.full((u.shape[0], 3), 255, dtype=np.uint8)

        # Decimate to cap max points
        M = u.shape[0]
        if M > self.proj_max_points:
            stride = max(1, M // self.proj_max_points)
            u = u[::stride]
            v = v[::stride]
            colors = colors[::stride]

        # Draw onto image
        # Convert to 24-bit RGB format for painting
        if qimg.format() != QtGui.QImage.Format_RGB888:
            qimg = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
        painter = QtGui.QPainter(qimg)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)

        # Group by color to reduce state changes; draw adjustable-sized squares for visibility
        if u.size > 0:
            s = max(1, int(getattr(self, 'proj_point_size', 3)))
            keys = (colors[:, 0].astype(np.int32) << 16) | (colors[:, 1].astype(np.int32) << 8) | (colors[:, 2].astype(np.int32))
            uniq = np.unique(keys)
            for key in uniq:
                mask = keys == key
                r = (key >> 16) & 0xFF
                g = (key >> 8) & 0xFF
                b = key & 0xFF
                color = QtGui.QColor(int(r), int(g), int(b))
                if s <= 1:
                    # Fast path: 1px points
                    pen = QtGui.QPen(color)
                    pen.setWidth(1)
                    painter.setPen(pen)
                    pts = [QtCore.QPoint(int(u[i]), int(v[i])) for i in np.nonzero(mask)[0]]
                    painter.drawPoints(QtGui.QPolygon(pts))
                else:
                    # Draw small filled squares for visibility
                    for i in np.nonzero(mask)[0]:
                        x = int(u[i]) - s // 2
                        y = int(v[i]) - s // 2
                        painter.fillRect(x, y, s, s, color)
        painter.end()
        return qimg
