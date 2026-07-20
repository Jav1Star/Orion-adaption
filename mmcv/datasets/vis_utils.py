import importlib.util
from pathlib import Path

import numpy as np


# 数据准备脚本依赖 Bench2Drive 官方工具函数，这里做最小桥接，避免再复制一份实现。
_UTILS_PATH = Path(__file__).resolve().parents[2] / "Bench2Drive" / "tools" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("bench2drive_tools_utils", _UTILS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Failed to load Bench2Drive tools utils from {_UTILS_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

edges = _MODULE.edges
DIS_CAR_SAVE = _MODULE.DIS_CAR_SAVE


def calculate_cube_vertices(center, extent):
    if isinstance(center, np.ndarray):
        cx, cy, cz = center.tolist()
    elif isinstance(center, (list, tuple)):
        cx, cy, cz = center
    else:
        cx, cy, cz = center.x, center.y, center.z

    if isinstance(extent, np.ndarray):
        x, y, z = extent.tolist()
    elif isinstance(extent, (list, tuple)):
        x, y, z = extent
    else:
        x, y, z = extent.x, extent.y, extent.z

    return [
        (cx + x, cy + y, cz + z),
        (cx + x, cy + y, cz - z),
        (cx + x, cy - y, cz + z),
        (cx + x, cy - y, cz - z),
        (cx - x, cy + y, cz + z),
        (cx - x, cy + y, cz - z),
        (cx - x, cy - y, cz + z),
        (cx - x, cy - y, cz - z),
    ]


def calculate_occlusion_stats(points, depths, depth_map, max_render_depth=100.0):
    # 数据准备只依赖可见/不可见/越界计数，这里按深度图做最小可见性判断。
    if points.size == 0 or depths.size == 0:
        return 0, 0, 0, []

    visible = 0
    invisible = 0
    outside = 0
    colored_points = []
    map_height, map_width = depth_map.shape[:2]

    for point, depth in zip(points, depths):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))

        if x < 0 or x >= map_width or y < 0 or y >= map_height:
            outside += 1
            continue

        rendered_depth = float(depth_map[y, x])
        if rendered_depth <= 0:
            invisible += 1
            continue

        rendered_depth = rendered_depth / 255.0 * max_render_depth
        if float(depth) <= rendered_depth + 1.0:
            visible += 1
            colored_points.append((x, y, (0, 255, 0)))
        else:
            invisible += 1
            colored_points.append((x, y, (0, 0, 255)))

    return visible, invisible, outside, colored_points
