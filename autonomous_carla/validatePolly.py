import os
import cv2
import numpy as np
import time
from process_image import KerasLaneDetector
from carla_interface import CarlaInterface
from polinomial.polyfit import fit_lanes_in_image, compute_virtual_centerline  # <-- Import the new function

# --- CONFIG ---
MODEL_PATH = 'models/lane_detector_combined_v2.keras'
CARLA_CONFIG = 'config/carla_config.yaml'
REFERENCE_SIZE = (640, 640)  # Match the size used in pure polyfit

# --- Initialize Lane Detector ---
detector = KerasLaneDetector(MODEL_PATH)

# --- Initialize Carla Interface ---
carla_interface = CarlaInterface(CARLA_CONFIG)
carla_interface.connect()
carla_interface.spawn_vehicle()
carla_interface.enable_autopilot(speed=8.0)  # Set autopilot with reduced speed (e.g., 8 km/h)

# --- Main loop ---
try:
	while True:
		# Get camera image from Carla
		image = carla_interface.get_camera_image()
		if image is None:
			print("No image received from Carla camera.")
			time.sleep(0.1)
			continue

		# Convert CARLA image to numpy RGB array
		if hasattr(image, 'raw_data'):
			array = np.frombuffer(image.raw_data, dtype=np.uint8)
			array = array.reshape((image.height, image.width, 4))
			rgb_array = array[:, :, :3] # BGR to RGB

		# Lane detection mask using make_predictions logic
		# Resize and normalize as in preprocess_image
		img_resized = cv2.resize(rgb_array, (detector.input_shape[1], detector.input_shape[0]))
		img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
		img_input = img_gray.astype(np.float32) / 255.0
		img_input = np.expand_dims(img_input, axis=-1)  # (H, W, 1)
		img_input = np.expand_dims(img_input, axis=0)   # (1, H, W, 1)
		# Predict mask as in predict_mask
		pred = detector.model.predict(img_input, verbose=0)[0]
		mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255
		# --- Resize mask to reference size for polyfit ---
		mask_polyfit = cv2.resize(mask, REFERENCE_SIZE)
		# Polyfit overlay
		lanes = fit_lanes_in_image(mask_polyfit)
		result = None
		if lanes:
			# Updated: pass both width and height to compute_virtual_centerline
			result = compute_virtual_centerline(lanes, REFERENCE_SIZE[0], REFERENCE_SIZE[1])

		# Prepare overlay in reference size
		if mask_polyfit.ndim == 2:
			overlay = cv2.cvtColor(mask_polyfit, cv2.COLOR_GRAY2BGR)
		else:
			overlay = mask_polyfit.copy()

		cv2.imshow('Lane Detection Overlay', overlay)
		# Draw polynomial curves on the overlay
		for lane in lanes:
			x_plot, y_plot = lane['curve']
			pts = np.vstack((x_plot, y_plot)).T.astype(np.int32)
			pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < overlay.shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < overlay.shape[0])]
			if len(pts) > 1:
				cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

		# Draw centerlines if available
		if result is not None:
			x_blend, y_blend, x_c1, x_c2 = result
			# Orange: blended centerline
			pts_blend = np.vstack((x_blend, y_blend)).T.astype(np.int32)
			pts_blend = pts_blend[(pts_blend[:, 0] >= 0) & (pts_blend[:, 0] < overlay.shape[1]) & (pts_blend[:, 1] >= 0) & (pts_blend[:, 1] < overlay.shape[0])]
			if len(pts_blend) > 1:
				cv2.polylines(overlay, [pts_blend], isClosed=False, color=(0, 140, 255), thickness=2)  # Orange (BGR)
			# Blue: c1 (original)
			# pts_c1 = np.vstack((x_c1, y_blend)).T.astype(np.int32)
			# pts_c1 = pts_c1[(pts_c1[:, 0] >= 0) & (pts_c1[:, 0] < overlay.shape[1]) & (pts_c1[:, 1] >= 0) & (pts_c1[:, 1] < overlay.shape[0])]
			# if len(pts_c1) > 1:
			# 	cv2.polylines(overlay, [pts_c1], isClosed=False, color=(255, 0, 0), thickness=2)  # Blue (BGR)
			# # Gray: c2 (car anchor)
			# pts_c2 = np.vstack((x_c2, y_blend)).T.astype(np.int32)
			# pts_c2 = pts_c2[(pts_c2[:, 0] >= 0) & (pts_c2[:, 0] < overlay.shape[1]) & (pts_c2[:, 1] >= 0) & (pts_c2[:, 1] < overlay.shape[0])]
			# if len(pts_c2) > 1:
			# 	cv2.polylines(overlay, [pts_c2], isClosed=False, color=(128, 128, 128), thickness=2)  # Gray (BGR)

		# --- Resize overlay back to display size ---
		overlay_display = cv2.resize(overlay, (mask.shape[1], mask.shape[0]))

		# Show the overlay in a separate OpenCV window

		# Show the overlay (model output + polyfit) in a separate window
		cv2.imshow('Model Output + Polyfit Overlay', overlay_display)

		# Show raw camera image in a separate OpenCV window
		cv2.imshow('Car Camera (Raw)', rgb_array)

		# Optional: press 'q' to quit (works for both windows)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

except KeyboardInterrupt:
	print("Validation interrupted by user.")
finally:
	carla_interface.cleanup()
	cv2.destroyAllWindows()
