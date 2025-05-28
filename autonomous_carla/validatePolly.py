import cv2
import numpy as np
import time
import tensorflow as tf
from process_image import KerasLaneDetector
from carla_interface import CarlaInterface
from polinomial.polyfit import fit_lanes_in_image, compute_virtual_centerline

# Configure GPU memory growth BEFORE any TensorFlow operations
def configure_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth configured for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

configure_gpu_memory()

# --- CONFIG ---
MODEL_PATH = 'models/lane_detector_combined_v2.keras'
CARLA_CONFIG = {
    'DEBUG': False,
    'synchronous_mode': True,
    'fixed_delta_seconds': 0.05  # Fixed syntax error
}
REFERENCE_SIZE = (640, 640)

class ValidationManager:
    def __init__(self):
        self.detector = KerasLaneDetector(MODEL_PATH)
        self.carla_interface = CarlaInterface(CARLA_CONFIG)
        self.validation_count = 0
        self.max_validations = 5  # Reduced for testing
        
    def initialize_carla(self):
        """Initialize CARLA connection."""
        self.carla_interface.connect()
        
    def run_single_validation(self):
        """Run a single validation session."""
        print(f"Iniciando validação {self.validation_count + 1}")
        
        try:
            self.carla_interface.spawn_vehicle()
            self.carla_interface.enable_autopilot(speed=8.0)
            
            validation_frames = 0
            max_frames = 500  # Reduced for testing
            
            while validation_frames < max_frames:
                try:
                    # Tick world and update spectator
                    self.carla_interface.tick_world()
                    self.carla_interface.update_spectator()
                    
                    # Get processed image directly
                    rgb_array = self.carla_interface.get_camera_image_as_array()
                    if rgb_array is None:
                        time.sleep(0.01)
                        continue
                    
                    # Process lane detection
                    processed_overlay = self._process_lane_detection(rgb_array)
                    
                    # Display images
                    cv2.imshow('Model Output + Polyfit Overlay', processed_overlay)
                    cv2.imshow('Car Camera (Raw)', rgb_array)
                    
                    # Handle user input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        return 'quit'
                    elif key == ord('n'):
                        return 'next'
                    elif key == ord('r'):
                        return 'reset'
                    
                    validation_frames += 1
                    
                except Exception as e:
                    print(f"Erro no frame {validation_frames}: {e}")
                    time.sleep(0.1)
                    continue
            
            return 'next'
            
        except Exception as e:
            print(f"Erro na validação: {e}")
            return 'reset'
    
    def _process_lane_detection(self, rgb_array):
        """Process lane detection on the given image."""
        # Lane detection logic (unchanged)
        img_resized = cv2.resize(rgb_array, (self.detector.input_shape[1], self.detector.input_shape[0]))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        img_input = img_gray.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)
        
        pred = self.detector.model.predict(img_input, verbose=0)[0]
        mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255
        mask_polyfit = cv2.resize(mask, REFERENCE_SIZE)
        
        lanes = fit_lanes_in_image(mask_polyfit)
        result = None
        
        if lanes:
            result = compute_virtual_centerline(lanes, REFERENCE_SIZE[0], REFERENCE_SIZE[1])
        
        if mask_polyfit.ndim == 2:
            overlay = cv2.cvtColor(mask_polyfit, cv2.COLOR_GRAY2BGR)
        else:
            overlay = mask_polyfit.copy()
        
        # Draw curves and centerlines (unchanged)
        for lane in lanes:
            x_plot, y_plot = lane['curve']
            pts = np.vstack((x_plot, y_plot)).T.astype(np.int32)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < overlay.shape[1]) & 
                     (pts[:, 1] >= 0) & (pts[:, 1] < overlay.shape[0])]
            if len(pts) > 1:
                cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
        
        if result is not None:
            x_blend, y_blend, x_c1, x_c2 = result
            pts_blend = np.vstack((x_blend, y_blend)).T.astype(np.int32)
            pts_blend = pts_blend[(pts_blend[:, 0] >= 0) & (pts_blend[:, 0] < overlay.shape[1]) & 
                                 (pts_blend[:, 1] >= 0) & (pts_blend[:, 1] < overlay.shape[0])]
            if len(pts_blend) > 1:
                cv2.polylines(overlay, [pts_blend], isClosed=False, color=(0, 140, 255), thickness=2)
        
        overlay_display = cv2.resize(overlay, (mask.shape[1], mask.shape[0]))
        return overlay_display
    
    def run_validation_loop(self):
        """Run the main validation loop."""
        try:
            while self.validation_count < self.max_validations:
                if self.validation_count > 0:
                    self.carla_interface.reset_world()
                
                result = self.run_single_validation()
                
                if result == 'quit':
                    break
                elif result == 'reset':
                    continue
                elif result == 'next':
                    self.validation_count += 1
                    if self.validation_count < self.max_validations:
                        time.sleep(1.0)
                    else:
                        print("Todas as validações concluídas!")
                        break
                        
        except KeyboardInterrupt:
            print("Validação interrompida pelo usuário.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.carla_interface.cleanup()
        cv2.destroyAllWindows()

def main():
    validator = ValidationManager()
    
    try:
        validator.initialize_carla()
        validator.run_validation_loop()
    except Exception as e:
        print(f"Erro: {e}")
        validator.cleanup()

if __name__ == "__main__":
    main()
