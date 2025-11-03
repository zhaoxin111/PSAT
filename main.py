# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QColor
from ui.mainwindow import Ui_MainWindow
from opengl_widget import GLWidget
from functools import partial
import sys
from utils.pointcloud import PointCloudReadThread
from utils.ground_filter import USE_CSF, GroundFilterThread
from utils.elevation import ElevationColorThread
from widgets.category_choice_dialog import CategoryChoiceDialog
from widgets.setting_dialog import SettingDialog
from widgets.about_dialog import AboutDialog
from widgets.shortcut_doalog import ShortCutDialog
from collections import OrderedDict
from configs import load_config, save_config, DEFAULT_CONFIG_FILE, MODE, DISPLAY
from json import load, dump
import functools
import numpy as np
import os
from typing import Optional


class Mainwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Mainwindow, self).__init__()
        self.setupUi(self)
        self.openGLWidget=GLWidget(self, self)
        self.setCentralWidget(self.openGLWidget)

        self.actionPick.setEnabled(False)
        self.actionCachePick.setEnabled(False)
        self.actionFilter.setEnabled(False)
        self.dockWidget_files.setVisible(False)
        #
        self.actionClassify.setVisible(False)
        self.actionGround_filter.setVisible(False)
        #

        self.config_file = DEFAULT_CONFIG_FILE
        self.current_file: Optional[str] = None
        self.current_root: Optional[str] = None
        self.results_root: Optional[str] = None
        self.save_state = True
        self.category_choice_dialog = CategoryChoiceDialog(self, self)
        self.setting_dialog = SettingDialog(self, self)
        self.shortcut_dialog = ShortCutDialog(self)
        self.about_dialog = AboutDialog(self)
        # Action: Reapply Category Colors
        self.actionReapplyColors = QtWidgets.QAction(QtGui.QIcon(":/icons/ui/icons/调色盘_platte.svg"), "Reapply Colors", self)
        self.actionReapplyColors.setObjectName("actionReapplyColors")
        self.actionReapplyColors.setToolTip("Clamp indices and refresh category colors")
        self.actionReapplyColors.triggered.connect(self.openGLWidget.reapply_category_colors)
        try:
            self.toolBar.addAction(self.actionReapplyColors)
        except Exception:
            pass
        try:
            self.menuView.addAction(self.actionReapplyColors)
        except Exception:
            pass

        self.message = QtWidgets.QLabel('')
        self.statusbar.addPermanentWidget(self.message)

        self.point_cloud_read_thread = PointCloudReadThread()
        self.point_cloud_read_thread.tag.connect(self.point_cloud_read_thread_finished)
        self.point_cloud_read_thread.message.connect(self.show_message)

        if USE_CSF:
            self.actionGround_filter.setVisible(True)
            self.ground_filter_thread = GroundFilterThread()
            self.ground_filter_thread.tag.connect(self.ground_filter_thread_finished)
            self.ground_filter_thread.message.connect(self.show_message)

        self.elevation_color_thread = ElevationColorThread()
        self.elevation_color_thread.tag.connect(self.elevation_color_thread_finished)
        self.elevation_color_thread.message.connect(self.show_message)

        self.category_color_dict: Optional[OrderedDict] = None

        self.trans = QtCore.QTranslator()

        # 初始化界面
        self.info_widget.setVisible(True)
        self.label_widget.setVisible(False)

        self.init_connect()
        self.reload_cfg()

        # Auto Save timer: saves when there are unsaved changes and auto-save is enabled
        self.auto_save_timer = QtCore.QTimer(self)
        self.auto_save_timer.setInterval(800)  # ms debounce
        self.auto_save_timer.timeout.connect(self._auto_save_tick)
        self.auto_save_timer.start()

        # Hotkeys: A = previous file, D = next file
        self.prev_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("A"), self)
        self.prev_shortcut.activated.connect(self.go_previous_pointcloud)
        self.next_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("D"), self)
        self.next_shortcut.activated.connect(self.go_next_pointcloud)

        # Pre-annotation defaults
        self.preanno_enabled = False
        self.preanno_radius = 0.4
        self.preanno_k = 3
        self._prev_annot_cache = None

    def open_file(self):
        self.dockWidget_files.setVisible(False)
        self.listWidget_files.clear()
        file, suffix = QtWidgets.QFileDialog.getOpenFileName(self, caption='point cloud file',
                                                             filter="point cloud (*.las *.ply *.txt)")
        if file:
            if not self.close_point_cloud():
                return
            self.current_root = os.path.split(file)[0]
            self.current_file = file
            self.point_cloud_read_thread_start(file)

    def open_folder(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dir:
            self.close_point_cloud()

            self.dockWidget_files.setVisible(True)
            self.listWidget_files.clear()

            self.current_root = dir
            file_list = [file for file in os.listdir(dir) if not file.endswith('.json')]
            for file in file_list:
                item = QtWidgets.QListWidgetItem()
                item.setSizeHint(QtCore.QSize(200, 30))
                item.setText(file)
                self.listWidget_files.addItem(item)

    def double_click_files_widget(self, item:QtWidgets.QListWidgetItem):
        # Capture current annotations before switching
        try:
            self._update_prev_cache()
        except Exception:
            pass
        file_path = os.path.join(self.current_root, item.text())
        self.current_file = file_path
        self.point_cloud_read_thread_start(file_path)

    def point_cloud_read_thread_start(self, file_path):
        if file_path.endswith('.las') or file_path.endswith('.ply') or file_path.endswith('.txt'):
            self.point_cloud_read_thread.set_file_path(file_path)   # 传递文件名
            self.point_cloud_read_thread.start()                    # 线程读取文件
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', "Only support file endwith '.txt', '.ply', '.las'")

    def point_cloud_read_thread_finished(self, tag:bool):
        if tag:
            pointcloud = self.point_cloud_read_thread.pointcloud
            if pointcloud is None:
                return
            #
            label_name = os.path.splitext(os.path.basename(self.current_file))[0] + '.json'
            candidate_files = []
            if getattr(self, 'results_root', None):
                candidate_files.append(os.path.join(self.results_root, label_name))
            candidate_files.append('.'.join(self.current_file.split('.')[:-1]) + '.json')
            categorys = None
            instances = None
            for label_file in candidate_files:
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        datas = load(f)
                        info = datas.get('info', '')
                        if info == 'PSAT label file.':
                            categorys = datas.get('categorys', [])
                            instances = datas.get('instances', [])
                            categorys = np.array(categorys, dtype=np.int16)
                            instances = np.array(instances, dtype=np.int16)

                            if categorys.shape[0] != pointcloud.xyz.shape[0] or instances.shape[0] != pointcloud.xyz.shape[0]:
                                QtWidgets.QMessageBox.warning(self, 'Warning', 'Point cloud size does not match label size!')
                                if categorys.shape[0] != pointcloud.xyz.shape[0]:
                                    categorys = None
                                if instances.shape[0] != pointcloud.xyz.shape[0]:
                                    instances = None
                            # Remap category indices from saved file to current config using index_category_dict
                            try:
                                index_category_dict = datas.get('index_category_dict', {})
                                # JSON keys may be strings; normalize to int
                                index_category_dict = {int(k): v for k, v in index_category_dict.items()}
                            except Exception:
                                index_category_dict = {}
                            if index_category_dict and categorys is not None:
                                current_categories = list(self.category_color_dict.keys())
                                uncls_index = current_categories.index('__unclassified__') if '__unclassified__' in current_categories else 0
                                # Build old_index -> new_index mapping via category name
                                remap = {
                                    old: (current_categories.index(cat) if cat in current_categories else uncls_index)
                                    for old, cat in index_category_dict.items()
                                }
                                # Apply remap to loaded category indices
                                categorys = np.array([remap.get(int(i), uncls_index) for i in categorys.tolist()], dtype=np.int16)
                            break
            if pointcloud.num_point < 1:
                return

            # If no labels found on disk and pre-annotation is enabled, try to propagate from previous frame
            if categorys is None and instances is None and getattr(self, 'preanno_enabled', False):
                try:
                    cats_pred, ins_pred, stats = self._preannotate_from_prev(pointcloud, self.preanno_radius, self.preanno_k)
                    if cats_pred is not None and ins_pred is not None:
                        categorys, instances = cats_pred, ins_pred
                        self.save_state = False  # dirty; produced pre-annotations
                        hit, total = stats.get('hit', 0), stats.get('total', 0)
                        used_r = stats.get('used_radius', self.preanno_radius)
                        mode = stats.get('mode', 'world')
                        try:
                            msg_r = f"{used_r:.3f}" if used_r is not None else "n/a"
                        except Exception:
                            msg_r = str(used_r)
                        self.show_message(f"Pre-annotate ({mode}): {hit}/{total} voted, r~{msg_r}, k={self.preanno_k}", 4000)
                except Exception as e:
                    self.show_message(f"Pre-annotate failed: {e}", 5000)

            self.openGLWidget.load_vertices(pointcloud, categorys, instances)
            #
            self.label_num_point.setText('{}'.format(pointcloud.num_point))
            self.label_size_x.setText('{:.2f}'.format(pointcloud.size[0]))
            self.label_size_y.setText('{:.2f}'.format(pointcloud.size[1]))
            self.label_size_z.setText('{:.2f}'.format(pointcloud.size[2]))
            self.label_offset_x.setText('{:.2f}'.format(pointcloud.offset[0]))
            self.label_offset_y.setText('{:.2f}'.format(pointcloud.offset[1]))
            self.label_offset_z.setText('{:.2f}'.format(pointcloud.offset[2]))

            self.setWindowTitle(pointcloud.file_path)
            self.actionPick.setEnabled(True)
            self.actionCachePick.setEnabled(True)
            self.actionFilter.setEnabled(True)

            # After loading current, update previous-frame annotation cache for next propagation
            try:
                self._update_prev_cache()
            except Exception:
                pass

            # Update projection overlay if enabled
            try:
                if getattr(self, 'actionToggleProjection', None) and self.actionToggleProjection.isChecked():
                    self.openGLWidget.update_projection_for_current(self.current_file)
            except Exception:
                pass

    def ground_filter_thread_start(self):
        if self.openGLWidget.pointcloud is None:
            return
        if self.openGLWidget.category_color is None:
            self.openGLWidget.category_color_update()

        self.ground_filter_thread.vertices = self.openGLWidget.pointcloud.xyz
        self.ground_filter_thread.start()

    def ground_filter_thread_finished(self, tag:bool):
        if tag:
            ground_index = self.ground_filter_thread.ground
            self.openGLWidget.categorys[ground_index] = 1
            self.openGLWidget.change_color_to_category()
            self.save_state = False

    def elevation_color_thread_start(self):
        if self.openGLWidget.pointcloud is None:
            return
        self.elevation_color_thread.vertices = self.openGLWidget.pointcloud.xyz
        self.elevation_color_thread.start()

    def elevation_color_thread_finished(self, tag):
        if tag:
            self.openGLWidget.elevation_color = self.elevation_color_thread.elevation_color
            self.openGLWidget.change_color_to_elevation()

    def classify_thread_start(self):
        if self.openGLWidget.pointcloud is None:
            return

        self.classify_thread.vertices = self.openGLWidget.pointcloud.xyz[self.openGLWidget.categorys!=1]    # 非地面点
        self.classify_thread.start()

    def classify_thread_finished(self, tag):
        if tag:
            seg, ins = self.classify_thread.seg, self.classify_thread.ins
            self.show_message("Classifier | Up sampling ...", 10000000)
            mask = self.openGLWidget.categorys!=1
            self.openGLWidget.categorys[mask] = seg
            self.openGLWidget.instances[mask] = ins
            self.openGLWidget.category_color_update()
            self.openGLWidget.instance_color_update()
            self.openGLWidget.change_color_to_category()
            self.save_state = False

    def close_point_cloud(self):
        if not self.save_state:
            result = QtWidgets.QMessageBox.question(self, 'Warning', 'Proceed without saved?', QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                return False
        self.current_file = None
        self.openGLWidget.reset()
        self.openGLWidget.update()
        self.message.clear()
        self.save_state = True

        self.setWindowTitle("PSAT - Point Cloud Segmentation Annotation Tool")
        self.info_widget.setVisible(True)
        self.label_num_point.setText('None')
        self.label_size_x.setText('None')
        self.label_size_y.setText('None')
        self.label_size_z.setText('None')
        self.label_offset_x.setText('None')
        self.label_offset_y.setText('None')
        self.label_offset_z.setText('None')
        self.label_widget.setVisible(False)

        self.actionPick.setEnabled(False)
        self.actionCachePick.setEnabled(False)
        return True

    def open_backgrpund_color_dialog(self):
        color = QtWidgets.QColorDialog.getColor(parent=self)
        if color.isValid():
            self.openGLWidget.set_background_color(color.redF(), color.greenF(), color.blueF())

    def show_message(self, message:str, msecs:int=5000):
        self.statusbar.showMessage(message, msecs)

    def reload_cfg(self):
        self.cfg = load_config(self.config_file)
        label_dict_list = self.cfg.get('label', [])
        d = OrderedDict()
        for index, label_dict in enumerate(label_dict_list):
            category = label_dict.get('name', '__unclassified__')
            color = label_dict.get('color', '#ffffff')
            d[category] = color
        self.category_color_dict = d

    def save_cfg(self, config_file):
        save_config(self.cfg, config_file)

    def save_category_and_instance(self):
        if self.openGLWidget.pointcloud is not None:
            if self.openGLWidget.categorys is not None and self.openGLWidget.instances is not None:
                if self.current_file is None:
                    return
                base_name = os.path.splitext(os.path.basename(self.current_file))[0] + '.json'
                target_dir = self.results_root if getattr(self, 'results_root', None) else os.path.dirname(self.current_file)
                os.makedirs(target_dir, exist_ok=True)
                label_file = os.path.join(target_dir, base_name)

                datas = {}
                datas['info'] = 'PSAT label file.'
                datas['point cloud file'] = self.current_file
                datas['index_category_dict'] = {index: category for index, (category, color) in enumerate(self.category_color_dict.items())}

                datas['categorys'] = self.openGLWidget.categorys.tolist()
                datas['instances'] = self.openGLWidget.instances.tolist()

                with open(label_file, 'w') as f:
                    dump(datas, f, indent=4)

                self.show_message('{} have saved!'.format(label_file))
                self.save_state = True

    def update_dock(self):
        if self.openGLWidget.display == DISPLAY.ELEVATION:
            self.info_widget.setVisible(True)
            self.label_widget.setVisible(False)
            self.dockWidget.setWindowTitle('Elevation')

            self.label_num_point.setText('{}'.format(self.openGLWidget.pointcloud.num_point))
            self.label_size_x.setText('{:.2f}'.format(self.openGLWidget.pointcloud.size[0]))
            self.label_size_y.setText('{:.2f}'.format(self.openGLWidget.pointcloud.size[1]))
            self.label_size_z.setText('{:.2f}'.format(self.openGLWidget.pointcloud.size[2]))
            self.label_offset_x.setText('{:.2f}'.format(self.openGLWidget.pointcloud.offset[0]))
            self.label_offset_y.setText('{:.2f}'.format(self.openGLWidget.pointcloud.offset[1]))
            self.label_offset_z.setText('{:.2f}'.format(self.openGLWidget.pointcloud.offset[2]))
        if self.openGLWidget.display == DISPLAY.RGB:
            self.info_widget.setVisible(True)
            self.label_widget.setVisible(False)
            self.dockWidget.setWindowTitle('RGB')

            self.label_num_point.setText('{}'.format(self.openGLWidget.pointcloud.num_point))
            self.label_size_x.setText('{:.2f}'.format(self.openGLWidget.pointcloud.size[0]))
            self.label_size_y.setText('{:.2f}'.format(self.openGLWidget.pointcloud.size[1]))
            self.label_size_z.setText('{:.2f}'.format(self.openGLWidget.pointcloud.size[2]))
            self.label_offset_x.setText('{:.2f}'.format(self.openGLWidget.pointcloud.offset[0]))
            self.label_offset_y.setText('{:.2f}'.format(self.openGLWidget.pointcloud.offset[1]))
            self.label_offset_z.setText('{:.2f}'.format(self.openGLWidget.pointcloud.offset[2]))

        if self.openGLWidget.display == DISPLAY.CATEGORY:
            self.info_widget.setVisible(False)
            self.label_widget.setVisible(True)
            self.dockWidget.setWindowTitle('Category')

            self.label_listWidget.clear()
            for index, (category, color) in enumerate(self.category_color_dict.items()):
                item = QtWidgets.QListWidgetItem()
                item.setSizeHint(QtCore.QSize(200, 30))
                widget = QtWidgets.QWidget()
                layout = QtWidgets.QHBoxLayout()
                layout.setContentsMargins(9, 1, 9, 1)

                check_box = QtWidgets.QCheckBox()
                check_box.setFixedWidth(20)
                check_box.setChecked(self.openGLWidget.category_display_state_dict.get(index, True))
                check_box.setObjectName('check_box')
                check_box.stateChanged.connect(functools.partial(self.point_cloud_visible))

                label_category = QtWidgets.QLabel()
                label_category.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                label_category.setText(category)
                label_category.setObjectName('label_category')

                label_color = QtWidgets.QLabel()
                label_color.setFixedWidth(10)
                label_color.setStyleSheet("background-color: {};".format(color))
                label_color.setObjectName('label_color')

                layout.addWidget(check_box)
                layout.addWidget(label_color)
                layout.addWidget(label_category)
                widget.setLayout(layout)

                self.label_listWidget.addItem(item)
                self.label_listWidget.setItemWidget(item, widget)

        if self.openGLWidget.display == DISPLAY.INSTANCE:
            self.info_widget.setVisible(False)
            self.label_widget.setVisible(True)
            self.dockWidget.setWindowTitle('Instance')

            self.label_listWidget.clear()
            instances_set = list(set(self.openGLWidget.instances.tolist()))
            instances_set.sort()
            color_map = (self.openGLWidget.color_map * 255).astype(np.int32)
            for index in instances_set:
                item = QtWidgets.QListWidgetItem()
                item.setSizeHint(QtCore.QSize(200, 30))
                widget = QtWidgets.QWidget()
                layout = QtWidgets.QHBoxLayout()
                layout.setContentsMargins(9, 1, 9, 1)

                check_box = QtWidgets.QCheckBox()
                check_box.setFixedWidth(20)
                check_box.setChecked(self.openGLWidget.instance_display_state_dict.get(index, True))
                check_box.setObjectName('check_box')
                check_box.stateChanged.connect(functools.partial(self.point_cloud_visible))

                label_instance = QtWidgets.QLabel()
                label_instance.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                label_instance.setText('{}'.format(index))
                label_instance.setObjectName('label_instance')

                c = color_map[index].tolist()
                color = QColor(int(c[0]), int(c[1]), int(c[2]))
                label_color = QtWidgets.QLabel()
                label_color.setFixedWidth(10)
                label_color.setStyleSheet("background-color: {};".format(color.name()))
                label_color.setObjectName('label_color')

                layout.addWidget(check_box)
                layout.addWidget(label_color)
                layout.addWidget(label_instance)
                widget.setLayout(layout)

                self.label_listWidget.addItem(item)
                self.label_listWidget.setItemWidget(item, widget)

    def point_cloud_visible(self):
        if self.openGLWidget.display == DISPLAY.CATEGORY:
            mask = np.ones(self.openGLWidget.categorys.shape, dtype=bool)
            for index in range(self.label_listWidget.count()):
                item = self.label_listWidget.item(index)
                widget = self.label_listWidget.itemWidget(item)
                check_box = widget.findChild(QtWidgets.QCheckBox, 'check_box')
                if not check_box.isChecked():
                    mask[self.openGLWidget.categorys==index] = False
                    self.openGLWidget.category_display_state_dict[index] = False
                else:
                    self.openGLWidget.category_display_state_dict[index] = True

            self.openGLWidget.mask = mask
            self.openGLWidget.current_vertices = self.openGLWidget.pointcloud.xyz[self.openGLWidget.mask]
            self.openGLWidget.current_colors = self.openGLWidget.category_color[self.openGLWidget.mask]

        elif self.openGLWidget.display == DISPLAY.INSTANCE:
            mask = np.ones(self.openGLWidget.instances.shape, dtype=bool)
            for index in range(self.label_listWidget.count()):
                item = self.label_listWidget.item(index)
                widget = self.label_listWidget.itemWidget(item)
                label_instance = widget.findChild(QtWidgets.QLabel, 'label_instance')
                label_instance = int(label_instance.text())
                check_box = widget.findChild(QtWidgets.QCheckBox, 'check_box')
                if not check_box.isChecked():
                    mask[self.openGLWidget.instances==label_instance] = False
                    self.openGLWidget.instance_display_state_dict[label_instance] = False
                else:
                    self.openGLWidget.instance_display_state_dict[label_instance] = True

            self.openGLWidget.mask = mask
            self.openGLWidget.current_vertices = self.openGLWidget.pointcloud.xyz[self.openGLWidget.mask]
            self.openGLWidget.current_colors = self.openGLWidget.instance_color[self.openGLWidget.mask]
        self.openGLWidget.init_vertex_vao()
        self.openGLWidget.update()

    def check_show_all(self):
        for index in range(self.label_listWidget.count()):
            item = self.label_listWidget.item(index)
            widget = self.label_listWidget.itemWidget(item)
            check_box = widget.findChild(QtWidgets.QCheckBox, 'check_box')
            check_box.setChecked(self.checkBox_showall.isChecked())

            self.openGLWidget.mask.fill(self.checkBox_showall.isChecked())
            self.openGLWidget.init_vertex_vao()
            self.openGLWidget.update()

    def setting(self):
        self.setting_dialog.load_cfg()
        self.setting_dialog.show()

    def translate(self, language='zh'):
        if language == 'zh':
            self.trans.load('ui/zh_CN')
        else:
            self.trans.load('ui/en')
        self.actionChinese.setChecked(language=='zh')
        self.actionEnglish.setChecked(language=='en')
        _app = QtWidgets.QApplication.instance()
        _app.installTranslator(self.trans)
        self.retranslateUi(self)
        self.category_choice_dialog.retranslateUi(self.category_choice_dialog)
        self.setting_dialog.retranslateUi(self.setting_dialog)
        self.about_dialog.retranslateUi(self.about_dialog)
        self.shortcut_dialog.retranslateUi(self.shortcut_dialog)

    def translate_to_chinese(self):
        self.translate('zh')
        self.cfg['language'] = 'zh'

    def translate_to_english(self):
        self.translate('en')
        self.cfg['language'] = 'en'

    def load_projection_calibration(self):
        """
        Select calibration.json (with camera_to_image, dist, lidar_to_camera) and optionally an image folder.
        Then update the projection overlay for the current point cloud if enabled.
        """
        try:
            json_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption='Select calibration.json', filter='JSON (*.json)')
            if not json_path:
                return
            ok = self.openGLWidget.load_projection_from_file(json_path)
            if not ok:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'Invalid calibration file or parse error.')
                return

            # If calibration didn’t define an image folder, ask user
            if not getattr(self.openGLWidget, 'proj_image_folder', None):
                img_dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption='Select image folder')
                if img_dir:
                    self.openGLWidget.proj_image_folder = img_dir

            # Default image extension if not set by calibration
            if not getattr(self.openGLWidget, 'proj_image_ext', None):
                self.openGLWidget.proj_image_ext = '.jpg'

            # If user has toggled overlay ON and a point cloud is loaded, update now
            if getattr(self, 'actionToggleProjection', None) and self.actionToggleProjection.isChecked():
                if self.current_file and os.path.isfile(self.current_file):
                    self.openGLWidget.update_projection_for_current(self.current_file)
                else:
                    self.show_message('Calibration loaded. Open a point cloud to see projection.', 4000)
        except Exception as e:
            self.show_message(f'Failed to load calibration: {e}', 5000)

    def shortcut(self):
        self.shortcut_dialog.show()

    def about(self):
        self.about_dialog.show()

    def on_actionAutoSave_toggled(self, checked: bool):
        # Update statusbar message when Auto Save toggled
        self.show_message(f"Auto Save {'enabled' if checked else 'disabled'}")

    def _auto_save_tick(self):
        # Periodically auto-save when dirty and auto-save is enabled
        try:
            if hasattr(self, 'actionAutoSave') and self.actionAutoSave.isChecked():
                if not self.save_state:
                    self.save_category_and_instance()
        except Exception:
            # Avoid any unexpected timer crash; do nothing
            pass

    def go_previous_pointcloud(self):
        self._navigate_pointcloud(-1)

    def go_next_pointcloud(self):
        self._navigate_pointcloud(1)

    def _navigate_pointcloud(self, direction: int):
        """
        Navigate to previous/next point cloud file within current_root.
        direction: -1 for previous (A), +1 for next (D)
        """
        if self.current_root is None:
            return
        # Capture current annotations before switching files
        try:
            self._update_prev_cache()
        except Exception:
            pass
        try:
            files = [f for f in os.listdir(self.current_root) if not f.endswith('.json')]
        except Exception:
            return
        if not files:
            return
        files.sort()
        if self.current_file is None:
            target_index = 0 if direction > 0 else len(files) - 1
        else:
            basename = os.path.basename(self.current_file)
            try:
                idx = files.index(basename)
            except ValueError:
                idx = 0
            target_index = (idx + direction) % len(files)

        target_name = files[target_index]
        target_path = os.path.join(self.current_root, target_name)

        # Reflect selection in list widget if present
        for i in range(self.listWidget_files.count()):
            item = self.listWidget_files.item(i)
            if item.text() == target_name:
                self.listWidget_files.setCurrentRow(i)
                break

        self.current_file = target_path
        self.point_cloud_read_thread_start(target_path)

    def _preannotate_from_prev(self, pointcloud, radius: float, k: int):
        """
        Pre-annotate current pointcloud using previous frame annotated points via voxel-hash radius-KNN voting.
        Strategy:
          1) Try world coordinates (xyz + offset) with progressive radii.
          2) If too few hits, fallback to local coordinates (xyz) with progressive radii.
        Returns (categorys, instances, stats dict).
        """
        import math
        prev = getattr(self, '_prev_annot_cache', None)
        if prev is None or prev.get('cats') is None or prev.get('ins') is None:
            # No cache available
            N0 = self.openGLWidget.pointcloud.num_point if self.openGLWidget.pointcloud is not None else 0
            return None, None, {'total': N0, 'hit': 0, 'used_radius': None, 'mode': None}

        prev_world = prev.get('xyz_world', prev.get('xyz'))
        prev_local = prev.get('xyz_local', None)
        prev_cats = prev['cats']
        prev_ins = prev['ins']

        # Prepare outputs
        N = pointcloud.xyz.shape[0]
        try:
            cats_list = list(self.category_color_dict.keys())
            uncls_index = cats_list.index('__unclassified__')
        except Exception:
            uncls_index = 0

        def run_with(prev_xyz: np.ndarray, curr_xyz: np.ndarray, base_radius: float):
            """
            Build voxel hash on prev_xyz, vote for labels of curr_xyz within radius, with progressive widening.
            Returns hits, used_radius, cats_out, ins_out.
            """
            cats_out = np.full(N, uncls_index, dtype=np.int16)
            ins_out = np.zeros(N, dtype=np.int16)

            def pass_once(rr: float):
                # Build buckets for prev_xyz
                rr = float(max(1e-6, rr))
                cell = rr * 0.75
                inv = 1.0 / cell

                def hkey(p):
                    return (int(math.floor(p[0] * inv)),
                            int(math.floor(p[1] * inv)),
                            int(math.floor(p[2] * inv)))

                buckets = {}
                for i in range(prev_xyz.shape[0]):
                    key = hkey(prev_xyz[i])
                    lst = buckets.get(key)
                    if lst is None:
                        buckets[key] = [i]
                    else:
                        lst.append(i)

                r_cells = int(math.ceil(rr / cell))
                nbr_offsets = [(dx, dy, dz)
                               for dx in range(-r_cells, r_cells + 1)
                               for dy in range(-r_cells, r_cells + 1)
                               for dz in range(-r_cells, r_cells + 1)]
                r2 = rr * rr
                kk = int(max(1, k))
                hits_local = 0

                for i in range(N):
                    p = curr_xyz[i]
                    base = hkey(p)
                    cand_idx = []
                    for off in nbr_offsets:
                        key = (base[0] + off[0], base[1] + off[1], base[2] + off[2])
                        if key in buckets:
                            cand_idx.extend(buckets[key])
                    if not cand_idx:
                        continue
                    d2 = np.sum((prev_xyz[cand_idx] - p) ** 2, axis=1)
                    within = np.where(d2 <= r2)[0]
                    if within.size == 0:
                        continue
                    sel = within[np.argsort(d2[within])[:kk]]
                    neigh_idx = np.asarray(cand_idx, dtype=np.int32)[sel]
                    cats = prev_cats[neigh_idx]
                    if cats.size == 1:
                        cat = int(cats[0])
                    else:
                        vals, counts = np.unique(cats, return_counts=True)
                        cat = int(vals[np.argmax(counts)])
                    cats_out[i] = cat
                    nearest = neigh_idx[np.argmin(d2[sel])]
                    ins_out[i] = int(prev_ins[nearest])
                    hits_local += 1
                return hits_local

            # Try widening radii
            used_r = None
            hits_total = 0
            for scale in (1.0, 2.0, 4.0, 8.0):
                rr = base_radius * scale
                hits_total = pass_once(rr)
                used_r = rr
                if hits_total >= max(1, int(0.01 * N)):
                    break
            return hits_total, used_r, cats_out, ins_out

        # Attempt world coordinates first (if available)
        best_mode = None
        best_hits = -1
        best_r = None
        best_cats = None
        best_ins = None

        if prev_world is not None:
            curr_world = pointcloud.xyz + pointcloud.offset
            hits_w, r_w, cats_w, ins_w = run_with(prev_world.astype(np.float32, copy=False),
                                                  curr_world.astype(np.float32, copy=False),
                                                  radius)
            best_mode, best_hits, best_r, best_cats, best_ins = 'world', hits_w, r_w, cats_w, ins_w

        # Fallback to local coordinates if coverage is too low
        if (best_hits < max(1, int(0.01 * N))) and (prev_local is not None):
            curr_local = pointcloud.xyz
            hits_l, r_l, cats_l, ins_l = run_with(prev_local.astype(np.float32, copy=False),
                                                  curr_local.astype(np.float32, copy=False),
                                                  radius)
            if hits_l >= best_hits:
                best_mode, best_hits, best_r, best_cats, best_ins = 'local', hits_l, r_l, cats_l, ins_l

        return best_cats, best_ins, {'total': N, 'hit': int(max(0, best_hits)), 'used_radius': best_r, 'mode': best_mode}

    def _update_prev_cache(self):
        """
        Capture current sample's annotated points into cache for next-frame pre-annotation.
        Store points that are annotated by category or have a non-zero instance id to be robust.
        """
        if self.openGLWidget.pointcloud is None or self.openGLWidget.categorys is None:
            self._prev_annot_cache = None
            return
        xyz_world = self.openGLWidget.pointcloud.xyz + self.openGLWidget.pointcloud.offset
        cats = self.openGLWidget.categorys
        ins = self.openGLWidget.instances if self.openGLWidget.instances is not None else np.zeros_like(cats, dtype=np.int16)
        try:
            cats_list = list(self.category_color_dict.keys())
            uncls_index = cats_list.index('__unclassified__')
        except Exception:
            uncls_index = 0
        # Consider annotated if category != unclassified OR instance != 0
        mask = (cats != uncls_index) | (ins != 0)
        if not np.any(mask):
            # still keep empty cache to avoid repeated work
            self._prev_annot_cache = {
                'xyz': np.empty((0,3), dtype=np.float32),         # kept for backward compatibility (world)
                'xyz_world': np.empty((0,3), dtype=np.float32),
                'xyz_local': np.empty((0,3), dtype=np.float32),
                'cats': np.empty((0,), dtype=np.int16),
                'ins': np.empty((0,), dtype=np.int16)
            }
            return
        xyz_world_masked = xyz_world[mask].astype(np.float32, copy=False)
        xyz_local_masked = self.openGLWidget.pointcloud.xyz[mask].astype(np.float32, copy=False)
        self._prev_annot_cache = {
            'xyz': xyz_world_masked,                 # kept for backward compatibility (world)
            'xyz_world': xyz_world_masked,
            'xyz_local': xyz_local_masked,
            'cats': cats[mask].astype(np.int16, copy=False),
            'ins': ins[mask].astype(np.int16, copy=False)
        }

    def init_connect(self):
        self.actionOpen.triggered.connect(self.open_file)
        self.actionOpenFolder.triggered.connect(self.open_folder)
        self.actionSetResultsFolder.triggered.connect(lambda: set_results_folder_for(self))
        self.listWidget_files.itemDoubleClicked.connect(self.double_click_files_widget)
        self.actionClose.triggered.connect(self.close_point_cloud)
        self.actionSave.triggered.connect(self.save_category_and_instance)
        self.actionExit.triggered.connect(self.close)

        self.actionPoint_size.triggered.connect(partial(self.openGLWidget.pointsize_change, 1))
        self.actionPoint_size_2.triggered.connect(partial(self.openGLWidget.pointsize_change, -1))
        self.actionTop_view.triggered.connect(self.openGLWidget.set_top_view)
        self.actionBottom_view.triggered.connect(self.openGLWidget.set_bottom_view)
        self.actionFront_view.triggered.connect(self.openGLWidget.set_front_view)
        self.actionBack_view.triggered.connect(self.openGLWidget.set_back_view)
        self.actionLeft_view.triggered.connect(self.openGLWidget.set_left_view)
        self.actionRight_view.triggered.connect(self.openGLWidget.set_right_view)

        self.actionBackground_color.triggered.connect(self.open_backgrpund_color_dialog)
        self.actionElevation.triggered.connect(self.openGLWidget.change_color_to_elevation)
        self.actionRgb.triggered.connect(self.openGLWidget.change_color_to_rgb)
        self.actionCategory.triggered.connect(self.openGLWidget.change_color_to_category)
        self.actionInstance.triggered.connect(self.openGLWidget.change_color_to_instance)
        self.actionGround_filter.triggered.connect(self.ground_filter_thread_start)
        self.actionClassify.triggered.connect(self.classify_thread_start)

        self.actionPick.triggered.connect(self.openGLWidget.change_mode_to_pick)
        self.actionCachePick.triggered.connect(self.openGLWidget.cache_pick)
        self.actionFilter.triggered.connect(self.openGLWidget.toggle_filter_mode)
        # Toggle enabling rotation-center-on-double-click
        self.actionRotationCenter.toggled.connect(self.openGLWidget.set_rotation_center_on_double_click_enabled)
        # Auto Save toggle UI
        self.actionAutoSave.toggled.connect(self.on_actionAutoSave_toggled)
        # Pre-annotation toggle
        if hasattr(self, 'actionPreAnnotate'):
            self.actionPreAnnotate.toggled.connect(lambda checked: setattr(self, 'preanno_enabled', checked))
        # Projection overlay actions
        if hasattr(self, 'actionToggleProjection'):
            self.actionToggleProjection.toggled.connect(self.openGLWidget.set_projection_enabled)
        if hasattr(self, 'actionLoadProjectionConfig'):
            self.actionLoadProjectionConfig.triggered.connect(self.load_projection_calibration)

        self.actionSetting.triggered.connect(self.setting)
        self.actionChinese.triggered.connect(self.translate_to_chinese)
        self.actionEnglish.triggered.connect(self.translate_to_english)
        self.actionShortcut.triggered.connect(self.shortcut)
        self.actionAbout.triggered.connect(self.about)

        self.checkBox_showall.stateChanged.connect(self.check_show_all)

        self.save_window_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("P"), self)
        self.save_window_shortcut.activated.connect(self.openGLWidget.save_windows_to_pic)


def set_results_folder_for(mainwindow: 'Mainwindow'):
    dir = QtWidgets.QFileDialog.getExistingDirectory(mainwindow, caption='Select results folder')
    if dir:
        mainwindow.results_root = dir
        mainwindow.show_message(f"Results folder set to: {dir}")
    else:
        # Cancelled; do nothing
        pass


if __name__ == '__main__':
    # Multiprocessing safety for PyInstaller/Windows
    try:
        from multiprocessing import freeze_support, set_start_method, get_start_method
        freeze_support()
        # Ensure 'spawn' start method; ignore if already set
        if get_start_method() != 'spawn':
            set_start_method('spawn', force=True)
    except Exception:
        # Swallow runtime errors when method already set or on non-Windows
        pass

    # Single-instance guard (socket lock). If another instance holds the port, exit quietly.
    import socket
    _single_instance_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Use a deterministic, app-specific localhost port
        _single_instance_sock.bind(('127.0.0.1', 49653))
    except OSError:
        # Another instance detected; avoid spawning additional windows
        sys.exit(0)

    # Normal GUI startup
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = Mainwindow()
    mainwindow.show()

    # Handle file path from argv if launched via file association or drag-drop
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        if os.path.isfile(candidate) and candidate.lower().endswith(('.las', '.ply', '.txt')):
            mainwindow.current_root = os.path.dirname(candidate)
            mainwindow.current_file = candidate
            mainwindow.point_cloud_read_thread_start(candidate)

    sys.exit(app.exec_())

