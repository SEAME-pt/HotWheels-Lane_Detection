#!/usr/bin/env python3
# make_predictions.py - Lane mask prediction post-processing script

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from models import get_custom_objects
import math

# ===== CONFIG =====
MODEL_PATH = "models/lane_detector.keras"
IMAGES_PATH = "images"
OUTPUT_PATH = "output"

TARGET_SIZE = (208, 208)
IS_GRAYSCALE = True
SHOW_ORIGINAL_IMAGE = False
SHOW_RAW_MASK = False
MIN_COMPONENT_SIZE = 350
MIN_COMPONENT_LENGTH = 260
DRAW_EXTREMITIES = False
SHOW_LANE_INFO = False
# ===== MERGE THRESHOLDS =====
ANGLE_THRESHOLD = 10.0                # in degrees
TARGET_DISTANCE_THRESHOLD = 10.0      # in pixels



def load_and_prepare_model():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    return load_model(MODEL_PATH, custom_objects=get_custom_objects())


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if IS_GRAYSCALE else cv2.IMREAD_COLOR)
    if img is None:
        return None, None, None
    original = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if IS_GRAYSCALE else img.copy()
    img_resized = cv2.resize(img, TARGET_SIZE)
    img_input = img_resized.astype(np.float32) / 255.0
    if IS_GRAYSCALE:
        img_input = np.expand_dims(img_input, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)
    return img, img_input, original


def predict_mask(model, img_input, original_shape):
    pred = model.predict(img_input, verbose=0)[0]
    mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (original_shape[1], original_shape[0]))

def get_extremities(contour):
    """
    Returns the two furthest-apart points of a contour.
    """
    max_dist = 0
    pt1, pt2 = contour[0][0], contour[0][0]
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            a = contour[i][0]
            b = contour[j][0]
            dist = np.linalg.norm(a - b)
            if dist > max_dist:
                max_dist = dist
                pt1, pt2 = a, b
    return (tuple(pt1), tuple(pt2))


def should_merge(lane1, lane2):
    angle_diff = abs(lane1["angle"] - lane2["angle"])
    angle_diff = min(angle_diff, 180 - angle_diff)

    if angle_diff < ANGLE_THRESHOLD:
        for pt1 in lane1["targets"]:
            for pt2 in lane2["targets"]:
                if np.linalg.norm(np.array(pt1) - np.array(pt2)) < TARGET_DISTANCE_THRESHOLD:
                    return True
    return False



def draw_lane_extremities(contour, canvas):
    """
    Draw red circles at the two furthest points of the contour.
    """
    if len(contour) < 2:
        return

    max_dist = 0
    ext_pt1, ext_pt2 = contour[0][0], contour[0][0]

    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            pt1 = contour[i][0]
            pt2 = contour[j][0]
            dist = np.linalg.norm(pt1 - pt2)
            if dist > max_dist:
                max_dist = dist
                ext_pt1, ext_pt2 = pt1, pt2

    cv2.circle(canvas, tuple(ext_pt1), 5, (0, 0, 255), -1)
    cv2.circle(canvas, tuple(ext_pt2), 5, (0, 0, 255), -1)

def get_extremities(contour):
        max_dist = 0
        ext1, ext2 = contour[0][0], contour[0][0]
        for i in range(len(contour)):
            for j in range(i + 1, len(contour)):
                pt1 = contour[i][0]
                pt2 = contour[j][0]
                dist = np.linalg.norm(pt1 - pt2)
                if dist > max_dist:
                    max_dist = dist
                    ext1, ext2 = pt1, pt2
        return ext1, ext2

def draw_connection_between_lanes(lane_info, canvas, color=(255, 255, 255), thickness=4):
    for i in range(len(lane_info)):
        for j in range(i + 1, len(lane_info)):
            lane1 = lane_info[i]
            lane2 = lane_info[j]

            if should_merge(lane1, lane2):
                cnt1 = lane1["contour"]
                cnt2 = lane2["contour"]

                ext1a, ext1b = get_extremities(cnt1)
                ext2a, ext2b = get_extremities(cnt2)

                # Find closest extremity pair
                pairs = [
                    (ext1a, ext2a),
                    (ext1a, ext2b),
                    (ext1b, ext2a),
                    (ext1b, ext2b)
                ]
                pt1, pt2 = min(pairs, key=lambda p: np.linalg.norm(np.array(p[0]) - np.array(p[1])))

                # Draw white line between extremities
                cv2.line(canvas, tuple(pt1), tuple(pt2), color, thickness)




def extract_lane_info(mask_resized, use_min_size=True, use_min_length=True):
    """
    Extracts lane segment info from a binary mask.
    Each segment includes:
        - angle (from line fit)
        - centroid (mass center)
        - targets: list of border intersection points
        - contour: original contour
    """
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lane_info = []
    height, width = mask_resized.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, closed=False)

        if (not use_min_size or area > MIN_COMPONENT_SIZE) and (not use_min_length or length > MIN_COMPONENT_LENGTH):
            if len(cnt) >= 2:
                [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()
                angle = math.degrees(math.atan2(vy, vx)) % 180

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue  # skip degenerate
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                intersections = []
                # Top border
                if vy != 0:
                    t = (0 - y0) / vy
                    x = x0 + vx * t
                    if 0 <= x < width:
                        intersections.append((int(x), 0))
                # Bottom border
                if vy != 0:
                    t = ((height - 1) - y0) / vy
                    x = x0 + vx * t
                    if 0 <= x < width:
                        intersections.append((int(x), height - 1))
                # Left border
                if vx != 0:
                    t = (0 - x0) / vx
                    y = y0 + vy * t
                    if 0 <= y < height:
                        intersections.append((0, int(y)))
                # Right border
                if vx != 0:
                    t = ((width - 1) - x0) / vx
                    y = y0 + vy * t
                    if 0 <= y < height:
                        intersections.append((width - 1, int(y)))

                if len(intersections) >= 2:
                    lane_info.append({
                        "index": len(lane_info) + 1,
                        "angle": angle,
                        "centroid": (cx, cy),
                        "targets": intersections[:2],
                        "contour": cnt
                    })

    return lane_info


def render_filtered_mask(lane_info, shape):
    """
    Renders the lane mask with optional lane numbers and extremities.
    Returns a BGR image.
    """
    mask_filtered = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    for lane in lane_info:
        contour = lane["contour"]
        cv2.drawContours(mask_filtered, [contour], -1, color=(255, 255, 255), thickness=cv2.FILLED)

        if DRAW_EXTREMITIES:
            draw_lane_extremities(contour, mask_filtered)

        if SHOW_LANE_INFO:
            cx, cy = lane["centroid"]
            cv2.putText(
                mask_filtered,
                f"{lane['index']}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255,),
                2,
                cv2.LINE_AA
            )

    return mask_filtered


def build_output_view(original, mask_resized, mask_filtered):
    views = []
    separator = np.ones((original.shape[0], 10, 3), dtype=np.uint8) * 255

    if SHOW_ORIGINAL_IMAGE:
        views.append(original)
        views.append(separator.copy())

    if SHOW_RAW_MASK:
        raw_mask = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        views.append(raw_mask)
        views.append(separator.copy())

    mask_filtered = cv2.cvtColor(mask_filtered, cv2.COLOR_BGR2RGB)
    views.append(mask_filtered)

    return np.hstack(views)



def create_info_panel(output_height, lane_info):
    panel_width = 300
    panel = np.ones((output_height, panel_width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    for lane in lane_info:
        idx = lane["index"]
        angle = lane["angle"]
        targets = lane["targets"]

        y_pos = 30 + (idx - 1) * 60
        cv2.putText(panel, f"Lane {idx}: {angle:.1f}°", (10, y_pos), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        for i, (tx, ty) in enumerate(targets):
            label = f"T{i+1}: ({tx}, {ty})"
            cv2.putText(panel, label, (10, y_pos + 20 + i * 20), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    return panel


def log_joinable_lanes(lane_info, img_name):
    joined = []
    for i in range(len(lane_info)):
        for j in range(i + 1, len(lane_info)):
            if should_merge(lane_info[i], lane_info[j]):
                joined.append((lane_info[i]["index"], lane_info[j]["index"]))

    if joined:
        print(f"\n🔗 Joinable lanes detected in {img_name}:")
        for a, b in joined:
            print(f"  → Lane {a} should be joined with Lane {b}")

def process_image(model, img_name):
    img_path = os.path.join(IMAGES_PATH, img_name)
    img, img_input, original = preprocess_image(img_path)
    if img is None:
        print(f"⚠️ Couldn't read {img_path}")
        return

    mask_resized = predict_mask(model, img_input, original.shape)

    lane_info_initial = extract_lane_info(mask_resized, use_min_length=False)

    if SHOW_LANE_INFO:
        log_joinable_lanes(lane_info_initial, img_name)

    mask_connected = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    draw_connection_between_lanes(lane_info_initial, mask_connected)

    mask_connected_gray = cv2.cvtColor(mask_connected, cv2.COLOR_BGR2GRAY)

    lane_info = extract_lane_info(mask_connected_gray, use_min_size=False)
    mask_filtered = render_filtered_mask(lane_info, original.shape)


    output = build_output_view(original, mask_resized, mask_filtered)

    save_path = os.path.join(OUTPUT_PATH, img_name)

    if SHOW_LANE_INFO:
        panel = create_info_panel(output.shape[0], lane_info)
        combined = np.hstack((output, panel))
        cv2.imwrite(save_path, combined)
    else:
        cv2.imwrite(save_path, output)



def main():
    model = load_and_prepare_model()
    for img_name in sorted(os.listdir(IMAGES_PATH)):
        if img_name.lower().endswith(".jpg"):
            process_image(model, img_name)
    print(f"✅ Done! Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
