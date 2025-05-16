import cv2
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline

def sliding_window_centroids(x, y, img_shape, num_windows=20):
    window_height = img_shape[0] // num_windows
    centroids_x, centroids_y = [], []

    for i in range(num_windows):
        win_y_low = img_shape[0] - (i + 1) * window_height
        win_y_high = img_shape[0] - i * window_height
        mask = (y >= win_y_low) & (y < win_y_high)
        x_window = x[mask]
        y_window = y[mask]
        if len(x_window) > 0:
            centroids_x.append(np.mean(x_window))
            centroids_y.append(np.mean(y_window))

    return np.array(centroids_y), np.array(centroids_x)

def spline_fit(y, x, s=20):
    if len(x) < 4:
        return None
    sort_idx = np.argsort(y)
    y_sorted = y[sort_idx]
    x_sorted = x[sort_idx]
    return UnivariateSpline(y_sorted, x_sorted, k=3, s=s)

def is_straight_line(y, x, threshold=0.995):
    if len(x) < 4: return False
    corr = np.corrcoef(x, y)[0, 1]
    return abs(corr) > threshold

def extract_center_curve_with_lanes(binary_mask):
    # Denoising
    kernel = np.ones((5, 5), np.uint8)
    img_clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Extract white pixels
    y_indices, x_indices = np.nonzero(img_clean)
    points = np.vstack((x_indices, y_indices)).T

    # Verificar se há pontos antes de chamar DBSCAN
    if points.shape[0] == 0:
        return None  # Retornar None se não houver pontos
    
    # DBSCAN Clustering
    db = DBSCAN(eps=25, min_samples=40).fit(points)
    labels = db.labels_
    unique_labels = set(labels) - {-1}

    spline_curves = []
    
    for label in unique_labels:
        mask = labels == label
        cluster_x = points[mask][:, 0]
        cluster_y = points[mask][:, 1]
        cent_y, cent_x = sliding_window_centroids(cluster_x, cluster_y, binary_mask.shape)
        
        if len(cent_y) < 4:
            continue
            
        # Hybrid fit
        if is_straight_line(cent_y, cent_x):
            coeffs = np.polyfit(cent_y, cent_x, deg=1)
            plot_y = np.linspace(min(cent_y), max(cent_y), 200)
            curve_x = np.polyval(coeffs, plot_y)
        else:
            spline = spline_fit(cent_y, cent_x, s=20)
            if spline is None:
                continue
            plot_y = np.linspace(min(cent_y), max(cent_y), 200)
            curve_x = spline(plot_y)
            
        in_bounds = (curve_x >= 0) & (curve_x < binary_mask.shape[1])
        
        if np.sum(in_bounds) < 10:
            continue
            
        spline_curves.append((np.mean(curve_x[in_bounds]), curve_x[in_bounds], plot_y[in_bounds]))
    
    # Sort lanes left to right
    spline_curves.sort(key=lambda item: item[0])
    
    # Compute center trajectory
    if len(spline_curves) >= 2:
        left_x = spline_curves[0][1]
        right_x = spline_curves[1][1]
        shared_y = spline_curves[0][2]
        min_len = min(len(left_x), len(right_x))
        center_curve = ((left_x[:min_len] + right_x[:min_len]) / 2, shared_y[:min_len])
    else:
        center_curve = None
    
    return center_curve, spline_curves  # (center_x, center_y) em coordenadas de imagem

if __name__ == "__main__": 
# --- Main ---
    if len(sys.argv) != 2:
        print("Usage: python script.py <binary_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    # Denoising
    kernel = np.ones((5, 5), np.uint8)
    img_clean = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Extract white pixels
    y_indices, x_indices = np.nonzero(img_clean)
    points = np.vstack((x_indices, y_indices)).T

    # DBSCAN Clustering
    db = DBSCAN(eps=25, min_samples=40).fit(points)
    labels = db.labels_
    unique_labels = set(labels) - {-1}

    colors = ['red', 'blue', 'yellow', 'purple', 'cyan']
    spline_curves = []

    # Fit per cluster using hybrid spline/linear
    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_x = points[mask][:, 0]
        cluster_y = points[mask][:, 1]

        cent_y, cent_x = sliding_window_centroids(cluster_x, cluster_y, img.shape)
        if len(cent_y) < 4:
            continue

        # Hybrid fit
        if is_straight_line(cent_y, cent_x):
            coeffs = np.polyfit(cent_y, cent_x, deg=1)
            plot_y = np.linspace(min(cent_y), max(cent_y), 200)
            curve_x = np.polyval(coeffs, plot_y)
        else:
            spline = spline_fit(cent_y, cent_x, s=20)
            if spline is None:
                continue
            plot_y = np.linspace(min(cent_y), max(cent_y), 200)
            curve_x = spline(plot_y)

        # Clip to image width
        in_bounds = (curve_x >= 0) & (curve_x < img.shape[1])
        if np.sum(in_bounds) < 10:
            continue

        spline_curves.append((
            np.mean(curve_x[in_bounds]),
            curve_x[in_bounds],
            plot_y[in_bounds],
            f'Lane {i+1}',
            colors[i % len(colors)]
        ))

    # Sort lanes left to right
    spline_curves.sort(key=lambda item: item[0])

    # Compute center trajectory
    center_curve = None
    if len(spline_curves) >= 2:
        left_x = spline_curves[0][1]
        right_x = spline_curves[1][1]
        shared_y = spline_curves[0][2]  # Assuming same Y range
        min_len = min(len(left_x), len(right_x))
        center_curve = ((left_x[:min_len] + right_x[:min_len]) / 2, shared_y[:min_len])

    # Plot
    plt.imshow(img, cmap='gray')
    for _, x_vals, y_vals, label, color in spline_curves:
        plt.plot(x_vals, y_vals, color=color, label=label)
    if center_curve is not None:
        plt.plot(center_curve[0], center_curve[1], 'g', label='Center trajectory')

    plt.axis('off')
    plt.legend(loc='upper right')
    plt.title("Spline+Linear Multi-Lane Detection")
    plt.show()
