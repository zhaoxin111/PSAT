import os
import sys
import yaml
from enum import Enum


def resource_path(relative_path: str) -> str:
    """
    Resolve resource path for both normal execution and PyInstaller bundles.
    - In a PyInstaller bundle, resources are extracted to sys._MEIPASS (onefile)
      or located next to the executable (onedir).
    - In normal execution, resolve relative to this file's directory.
    """
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


DEFAULT_CONFIG_FILE = resource_path('default.yaml')

def load_config(file):
    with open(file, 'rb')as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg

def save_config(cfg, file):
    s = yaml.dump(cfg)
    with open(file, 'w') as f:
        f.write(s)
    return True

class MODE(Enum):
    DRAW_MODE = 0
    VIEW_MODE = 1

class DISPLAY(Enum):
    ELEVATION = 0
    RGB = 1
    CATEGORY = 2
    INSTANCE = 3
