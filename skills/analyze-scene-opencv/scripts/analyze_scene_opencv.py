#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python", "numpy"]
# ///

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def classify_color_label(hue: float) -> str:
    """根据主色调做保守粗分类，不直接输出 Apple/Tomato 等具体类别。"""
    if hue < 10 or hue >= 170:
        return "red_small_object"
    if 10 <= hue < 25:
        return "orange_small_object"
    if 25 <= hue < 40:
        return "yellow_small_object"
    if 40 <= hue < 85:
        return "green_small_object"
    if 85 <= hue < 130:
        return "blue_small_object"
    return "unknown_small_object"


def safe_mean_hue(hsv_roi: np.ndarray, mask: np.ndarray) -> float:
    pixels = hsv_roi[:, :, 0][mask > 0]
    if pixels.size == 0:
        return 0.0
    return float(np.mean(pixels))


def contour_center(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    return x + w // 2, y + h // 2


def infer_position_hint(cx: int, roi_w: int) -> str:
    """基于图像水平位置给出粗略方位。"""
    if cx < roi_w * 0.33:
        return "front-left"
    if cx > roi_w * 0.66:
        return "front-right"
    return "front-center"


def local_background_contrast(
    hsv_image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    object_mask: np.ndarray,
    pad: int = 14,
) -> tuple[float, float]:
    """Compare object saturation/value with its surrounding local background."""
    img_h, img_w = hsv_image.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w, x + w + pad)
    y1 = min(img_h, y + h + pad)

    obj_roi = hsv_image[y:y + h, x:x + w]
    obj_pixels = obj_roi[object_mask > 0]
    if obj_pixels.size == 0:
        return 0.0, 0.0

    local_roi = hsv_image[y0:y1, x0:x1]
    ring_mask = np.ones((y1 - y0, x1 - x0), dtype=np.uint8)
    ring_mask[y - y0:y - y0 + h, x - x0:x - x0 + w] = 0
    ring_pixels = local_roi[ring_mask > 0]
    if ring_pixels.size == 0:
        return 255.0, 255.0

    sat_contrast = abs(float(np.mean(obj_pixels[:, 1])) - float(np.mean(ring_pixels[:, 1])))
    value_contrast = abs(float(np.mean(obj_pixels[:, 2])) - float(np.mean(ring_pixels[:, 2])))
    return sat_contrast, value_contrast


def build_error(message: str) -> None:
    print(json.dumps({"status": "error", "message": message}, ensure_ascii=False))
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scene image with OpenCV for robot-cleaner V1")
    parser.add_argument("--image", required=True, help="Absolute path to image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        build_error(f"Image not found: {str(image_path)}")

    # cv2.imread can fail on Windows paths that contain non-ASCII
    # characters. np.fromfile + cv2.imdecode handles those paths reliably.
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR) if image_bytes.size else None
    if image is None:
        build_error(f"Failed to read image: {str(image_path)}")

    h, w = image.shape[:2]

    # 只分析更靠近地面的底部区域。当前机器人默认 LookDown，因此底部 ROI 更符合地面视野。
    floor_y0 = int(h * 0.60)
    floor_roi = image[floor_y0:h, :]

    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # 1. 垃圾候选检测：颜色阈值 + 轮廓过滤
    # ------------------------------------------------------------------
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    red_mask = np.zeros_like(sat, dtype=np.uint8)
    red_mask[((hue < 12) | (hue > 170)) & (sat > 35) & (val > 25)] = 255

    # Do not merge every saturated pixel into one mask: wood floors are often
    # orange/brown and can swallow a real target into a huge discarded contour.
    color_masks = []
    color_masks.append(red_mask)

    orange_mask = np.zeros_like(sat, dtype=np.uint8)
    orange_mask[(hue >= 12) & (hue < 25) & (sat > 85) & (val > 60)] = 255
    color_masks.append(orange_mask)

    yellow_mask = np.zeros_like(sat, dtype=np.uint8)
    yellow_mask[(hue >= 25) & (hue < 40) & (sat > 55) & (val > 40)] = 255
    color_masks.append(yellow_mask)

    green_mask = np.zeros_like(sat, dtype=np.uint8)
    green_mask[(hue >= 40) & (hue < 85) & (sat > 55) & (val > 40)] = 255
    color_masks.append(green_mask)

    blue_mask = np.zeros_like(sat, dtype=np.uint8)
    blue_mask[(hue >= 85) & (hue < 130) & (sat > 55) & (val > 40)] = 255
    color_masks.append(blue_mask)

    contours = []
    for color_mask in color_masks:
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(mask_contours)

    trash_candidates = []
    ignored_candidates = []

    roi_h, roi_w = floor_roi.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)#轮廓的实际面积
        if area < 60 or area > 9000:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)#外接矩形

        if bw <= 3 or bh <= 3:
            continue

        extent = area / float(max(bw * bh, 1))
        if extent < 0.45:
            continue

        # 极端细长物体更可能是边缘/纹理噪声。
        if bw / max(bh, 1) > 6.0 or bh / max(bw, 1) > 6.0:
            continue

        mask_local = np.zeros((bh, bw), dtype=np.uint8)
        cnt_shifted = cnt - np.array([[[x, y]]])
        cv2.drawContours(mask_local, [cnt_shifted], -1, 255, thickness=-1)

        hsv_local = hsv[y:y + bh, x:x + bw]
        mean_hue = safe_mean_hue(hsv_local, mask_local)
        label = classify_color_label(mean_hue)

        cx, cy = contour_center(x, y, bw, bh)
        full_y = y + floor_y0
        full_bottom = full_y + bh

        position_hint = infer_position_hint(cx, roi_w)

        # 地面判定：要求目标中心和底边都位于画面下部，降低桌面/台面误清风险。
        center_y_ratio = (full_y + bh / 2) / max(h, 1)
        bottom_y_ratio = full_bottom / max(h, 1)
        is_floor_level = (bottom_y_ratio >= 0.72) and (center_y_ratio >= 0.63)

        sat_contrast, value_contrast = local_background_contrast(
            hsv_full,
            x,
            full_y,
            bw,
            bh,
            mask_local,
        )

        # Yellow/orange wood floors and bright light patches can satisfy the
        # bottom-of-image rule. If they barely differ from the surrounding
        # floor in saturation, treat them as floor texture rather than trash.
        touches_image_bottom = full_bottom >= h - 2
        large_warm_floor_patch = (
            label in {"yellow_small_object", "orange_small_object"}
            and area > 1200
            and (value_contrast < 30.0 or touches_image_bottom)
        )
        floor_like_texture = (
            is_floor_level
            and label in {"yellow_small_object", "orange_small_object"}
            and (sat_contrast < 20.0 or large_warm_floor_patch)
        )

        surface_hint = (
            "floor_like_texture_or_light_patch"
            if floor_like_texture
            else ("floor" if is_floor_level else "countertop_or_nonfloor")
        )

        # V1 关键语义：
        # reachable：理论上可通过对位后处理；
        # cleanable_now：当前不再移动，直接调用 clean-garbage 是合理的；
        # needs_alignment：目标在前左/前右，需要先转向对位，再复核。
        reachable = (
            is_floor_level
            and not floor_like_texture
            and position_hint in {"front-left", "front-center", "front-right"}
        )
        cleanable_now = is_floor_level and not floor_like_texture and position_hint == "front-center"
        needs_alignment = (
            is_floor_level
            and not floor_like_texture
            and position_hint in {"front-left", "front-right"}
        )

        candidate = {
            "label": label,
            "bbox": {
                "x": int(x),
                "y": int(full_y),
                "w": int(bw),
                "h": int(bh),
            },
            "center": {
                "x": int(cx),
                "y": int(full_y + bh // 2),
            },
            "position_hint": position_hint,
            "reachable": bool(reachable),
            "surface_hint": surface_hint,
            "is_floor_level": bool(is_floor_level),
            "cleanable_now": bool(cleanable_now),
            "needs_alignment": bool(needs_alignment),
            "area": float(area),
            "center_y_ratio": round(float(center_y_ratio), 3),
            "bottom_y_ratio": round(float(bottom_y_ratio), 3),
            "sat_contrast": round(float(sat_contrast), 2),
            "value_contrast": round(float(value_contrast), 2),
            "touches_image_bottom": bool(touches_image_bottom),
        }

        if reachable:
            trash_candidates.append(candidate)
        else:
            ignored_candidates.append(candidate)

    # 按直接可清优先，其次面积大者优先。
    trash_candidates.sort(key=lambda c: (not c["cleanable_now"], -c["area"]))

    floor_trash_detected = len(trash_candidates) > 0
    direct_cleanable_detected = any(c["cleanable_now"] for c in trash_candidates)
    alignment_needed = any(c["needs_alignment"] for c in trash_candidates)

    # ------------------------------------------------------------------
    # 2. 障碍与可探索方向估计
    # ------------------------------------------------------------------
    edges = cv2.Canny(gray, 70, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    sectors = {
        "left": edges[:, : roi_w // 3],
        "forward": edges[:, roi_w // 3: 2 * roi_w // 3],
        "right": edges[:, 2 * roi_w // 3:],
    }

    occupancy = {}
    for name, sector in sectors.items():
        ratio = float(np.count_nonzero(sector)) / float(sector.size)
        occupancy[name] = ratio

    open_directions = [name for name, ratio in occupancy.items() if ratio < 0.16]
    obstacle_ahead = occupancy["forward"] >= 0.16
    frontier_exists = len(open_directions) > 0
    floor_clean = not floor_trash_detected

    confidence = 0.60
    if floor_trash_detected:
        confidence += 0.15
    if direct_cleanable_detected:
        confidence += 0.05
    if frontier_exists:
        confidence += 0.10
    confidence = min(confidence, 0.95)

    # 给 Agent 一个保守建议。最终动作仍由 Agent 决策。
    recommended_action = "none"
    if direct_cleanable_detected:
        recommended_action = "clean-garbage"
    elif alignment_needed:
        first_align = next(c for c in trash_candidates if c["needs_alignment"])
        if first_align["position_hint"] == "front-left":
            recommended_action = "RotateLeft"
        elif first_align["position_hint"] == "front-right":
            recommended_action = "RotateRight"
    elif obstacle_ahead:
        if "left" in open_directions:
            recommended_action = "RotateLeft"
        elif "right" in open_directions:
            recommended_action = "RotateRight"
        else:
            recommended_action = "MoveBack"
    elif frontier_exists and "forward" in open_directions:
        recommended_action = "MoveAhead"
    elif frontier_exists:
        recommended_action = "RotateLeft" if "left" in open_directions else "RotateRight"

    notes = []
    if direct_cleanable_detected:
        notes.append("Detected front-center floor-level target. Direct cleaning is allowed by V1 rule.")
    elif alignment_needed:
        notes.append("Detected floor-level target, but it is not centered. Alignment is required before cleaning.")
    elif floor_trash_detected:
        notes.append("Detected floor-level candidate, but it is not directly cleanable now.")
    else:
        notes.append("No obvious cleanable floor-level trash detected in current frame.")

    if ignored_candidates:
        notes.append(
            f"Ignored {len(ignored_candidates)} non-floor or uncertain targets "
            f"(likely countertop / tabletop / elevated objects)."
        )

    if obstacle_ahead:
        notes.append("Forward direction appears visually cluttered; MoveAhead should be conservative.")
    else:
        notes.append("Forward direction appears relatively open.")

    result = {
        "status": "success",
        "image_path": str(image_path),
        "floor_trash_detected": floor_trash_detected,
        "direct_cleanable_detected": direct_cleanable_detected,
        "alignment_needed": alignment_needed,
        "trash_candidates": trash_candidates,
        "ignored_candidates": ignored_candidates,
        "obstacle_ahead": obstacle_ahead,
        "open_directions": open_directions,
        "frontier_exists": frontier_exists,
        "floor_clean": floor_clean,
        "analysis_confidence": round(confidence, 2),
        "occupancy": {k: round(v, 3) for k, v in occupancy.items()},
        "recommended_action": recommended_action,
        "notes": notes,
    }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
