import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline
import warnings

warnings.simplefilter('ignore', np.RankWarning)

def load_images_from_folder(folder_path, extensions=('.png', '.jpg', '.jpeg')):
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(extensions)])
    images = []

    for filename in image_files:
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((filename, img))
    return images

def extract_lane_points(img):
    y, x = np.nonzero(img)
    return np.vstack((x, y)).T

def cluster_lane_points(points, eps=5, min_samples=20):
    if len(points) == 0:
        return np.array([]), []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return clustering.labels_, list(set(clustering.labels_) - {-1})

def sliding_window_centroids(x, y, img_shape, num_windows=20, smooth=False):
    h = img_shape[0] // num_windows
    cx, cy = [], []

    for i in range(num_windows):
        y_low, y_high = img_shape[0] - (i + 1) * h, img_shape[0] - i * h
        mask = (y >= y_low) & (y < y_high)
        if np.any(mask):
            cx.append(np.mean(x[mask]))
            cy.append(np.mean(y[mask]))

    cx, cy = np.array(cx), np.array(cy)

    if smooth and len(cx) >= 3:
        cx[1:-1] = (cx[:-2] + cx[1:-1] + cx[2:]) / 3

    return cy, cx

def has_sign_flip(curve):
    second_deriv = np.gradient(np.gradient(curve))
    return np.any(np.sign(second_deriv[1:]) != np.sign(second_deriv[:-1]))

def is_straight_line(y, x, threshold=0.98):
    return len(x) >= 4 and abs(np.corrcoef(x, y)[0, 1]) > threshold

def fit_lane_curve(y, x, img_width, y_plot):
    if is_straight_line(y, x):
        coeffs = np.polyfit(y, x, 1)
        return np.polyval(coeffs, y_plot)

    coeffs = np.polyfit(y, x, 2)
    a = coeffs[0]
    if abs(a) > 0.0012 and len(y) >= 4:
        s = 10 + len(x) * 0.3
        spline = UnivariateSpline(y, x, k=3, s=s)
        return spline(y_plot)
    return np.polyval(coeffs, y_plot)

def fit_lanes_in_image(img):
    points = extract_lane_points(img)
    labels, unique_labels = cluster_lane_points(points)

    lanes = []
    for label in unique_labels:
        mask = labels == label
        x, y = points[mask][:, 0], points[mask][:, 1]

        cent_y, cent_x = sliding_window_centroids(x, y, img.shape, smooth=False)
        if len(cent_y) < 2:
            continue

        sorted_idx = np.argsort(cent_y)
        cent_y, cent_x = cent_y[sorted_idx], cent_x[sorted_idx]

        try:
            test_curve = np.polyval(np.polyfit(cent_y, cent_x, 2), cent_y)
            if has_sign_flip(test_curve):
                cent_y, cent_x = sliding_window_centroids(x, y, img.shape, smooth=True)
                sorted_idx = np.argsort(cent_y)
                cent_y, cent_x = cent_y[sorted_idx], cent_x[sorted_idx]
        except np.RankWarning:
            continue

        y_min, y_max = np.min(cent_y), np.max(cent_y)
        y_plot = np.linspace(max(0, y_min - 30), min(img.shape[0], y_max + 10), 300)
        x_plot = fit_lane_curve(cent_y, cent_x, img.shape[1], y_plot)

        lanes.append({
            'centroids': (cent_x, cent_y),
            'curve': (x_plot, y_plot)
        })
    return lanes

def select_relevant_lanes(lanes, img_width, img_height, filename=None):
    img_center = img_width / 2
    left_lane = None
    right_lane = None

    lane_infos = []

    for lane in lanes:
        x_plot, y_plot = lane['curve']

        # Filter lanes with points only in bottom half of the image
        bottom_half_mask = y_plot >= (img_height / 2)
        if np.any(bottom_half_mask):
            avg_x_bottom = np.mean(x_plot[bottom_half_mask])
            lane_infos.append((avg_x_bottom, lane))

    # Sort lanes left to right
    lane_infos.sort(key=lambda t: t[0])

    for avg_x, lane in lane_infos:
        if avg_x < img_center:
            left_lane = lane
        elif avg_x >= img_center and right_lane is None:
            right_lane = lane
            break

    return left_lane, right_lane


def compute_virtual_centerline(lanes, img_width, img_height, lane_width_px=300):
    apply_blending = True
    left_lane, right_lane = select_relevant_lanes(lanes, img_width, img_height)
    car_x = img_width // 2  # Car's horizontal position

    if left_lane and right_lane:
        # --- Midpoint Method ---
        x_left, y_left = left_lane['curve']
        x_right, y_right = right_lane['curve']

        y_min = max(np.min(y_left), np.min(y_right))
        y_start = img_height - 1
        y_common = np.linspace(y_start, y_min, 300)  # from bottom (car) to top

        # Interpolate with extrapolation for extension
        x_left_interp = np.interp(y_common, y_left, x_left, left=x_left[0], right=x_left[-1])
        x_right_interp = np.interp(y_common, y_right, x_right, left=x_right[0], right=x_right[-1])
        x_c1 = (x_left_interp + x_right_interp) / 2  # Original centerline

        x_c2 = np.full_like(y_common, car_x)  # Car-centered horizontal line

        if not apply_blending:
            print("⚠️ Centerline: no blending applied.")
            return x_c1, y_common, x_c1, x_c2

        # Blending: full c2 at bottom, full c1 at top
        w = (y_common[0] - y_common) / (y_common[0] - y_common[-1])
        x_blend = w * x_c1 + (1 - w) * x_c2

        return x_blend, y_common, x_c1, x_c2

    elif left_lane or right_lane:
        # --- Offset Method ---
        lane = left_lane if left_lane else right_lane
        x_lane, y_lane = lane['curve']

        direction = 1 if left_lane else -1
        x_c1 = x_lane + direction * lane_width_px / 2  # Offset original curve

        y_min = np.min(y_lane)
        y_start = img_height - 1
        y_common = np.linspace(y_start, y_min, 300)  # from bottom to top

        x_c1_interp = np.interp(y_common, y_lane, x_c1, left=x_c1[0], right=x_c1[-1])
        x_c2 = np.full_like(y_common, car_x)

        if not apply_blending:
            print("⚠️ Centerline: no blending applied.")
            return x_c1_interp, y_common, x_c1_interp, x_c2

        # Blending: full c2 at bottom, full c1 at top
        w = (y_common[0] - y_common) / (y_common[0] - y_common[-1])
        x_blend = w * x_c1_interp + (1 - w) * x_c2

        return x_blend, y_common, x_c1_interp, x_c2

    else:
        return None

def plot_lanes_on_ax(ax, img, lanes, colors=None, img_name=None):
    if colors is None:
        colors = ['red', 'blue', 'yellow', 'purple', 'green']

    ax.imshow(img, cmap='gray')
    ax.axis('off')

    display_polyfit = True
    if not display_polyfit:
        return
    for i, lane in enumerate(lanes):
        cx, cy = lane['centroids']
        x_plot, y_plot = lane['curve']
        in_bounds = (x_plot >= 0) & (x_plot < img.shape[1])

        ax.plot(x_plot[in_bounds], y_plot[in_bounds],
                color=colors[i % len(colors)], linewidth=2)
        ax.scatter(cx, cy, color="green", s=8, marker='o',
                   edgecolors='k', zorder=10)

def display_images_with_polyfit(images, cols=4):
    num_images = len(images)
    rows = (num_images + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axs = axs.flatten()

    for idx, (filename, img) in enumerate(images):
        ax = axs[idx]
        ax.set_title(filename, fontsize=8)

        lanes = fit_lanes_in_image(img)
        plot_lanes_on_ax(ax, img, lanes, img_name=filename)
        # Now compute and plot centerline
        display_centerline = True
        display_c1_c2 = False
        if not display_centerline:
            continue
        result = compute_virtual_centerline(lanes, img.shape[1], img.shape[0])
        if result is not None:
            x_blend, y_blend, x_c1, x_c2 = result

            if display_c1_c2:
                ax.plot(x_blend, y_blend, color='orange', label='blended centerline', linewidth=2)
                ax.plot(x_c1, y_blend, color='blue', linestyle='--', label='c1 (original)', linewidth=1)
                ax.plot(x_c2, y_blend, color='gray', linestyle=':', label='c2 (car anchor)', linewidth=1)
            else:
                ax.plot(x_blend, y_blend, color='orange', label='blended centerline', linestyle='--', linewidth=2)


    for j in range(len(images), len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    images = load_images_from_folder("frames")
    if images:
        display_images_with_polyfit(images)

if __name__ == "__main__":
    main()
