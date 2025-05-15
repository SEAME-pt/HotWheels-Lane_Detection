import cv2
import numpy as np
import os
import pygame
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

# import tensorflow as tf
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
print("Versão do TensorFlow:", tf.__version__)
print("Versão do CUDA:", tf.sysconfig.get_build_info()["cuda_version"])
print("Versão do cuDNN:", tf.sysconfig.get_build_info()["cudnn_version"])

# Import custom objects for model loading
from models import get_custom_objects

class KerasLaneDetector:
    def __init__(self, model_path='lane_detector_final'):
        """Initialize the lane detector with a Keras model."""
        # Load custom objects dictionary
        custom_objects = get_custom_objects()
        
        # Load the Keras model
        try:
            self.model = load_model(model_path, custom_objects=custom_objects)
            print(f"✅ Lane detection model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Set to evaluation mode
        self.model.trainable = False
        
        # Define input image size based on the model's input layer
        self.input_shape = self.model.input_shape[1:3]  # (height, width)
        print(f"Model input shape: {self.input_shape}")

    def sharpen_image(self, image):
        """Applies unsharp masking to sharpen the image."""
        # Convert to grayscale if needed
        blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return sharpened

    def preprocess_image(self, input_image):
        """Preprocess the image for the Keras model."""
        # Convert to grayscale if input is color
        if len(input_image.shape) == 3 and input_image.shape[2] > 1:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to match model input dimensions
        input_image = cv2.resize(input_image, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize to [0, 1] range
        input_image = input_image.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        input_image = np.expand_dims(input_image, axis=-1)  # Add channel dimension
        input_image = np.expand_dims(input_image, axis=0)   # Add batch dimension
        
        return input_image

    def predict(self, input_tensor):
        """Run inference with the Keras model."""
        # Pass through the model
        prediction = self.model.predict(input_tensor, verbose=0)
        
        # Convert to binary segmentation mask
        binary_seg = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        
        return binary_seg

    def process_and_visualize(self, input_image_path, save_dir="lane_output"):
        """Processes the image through the lane detection model and returns a Pygame surface."""
        os.makedirs(save_dir, exist_ok=True)

        # Load input image
        input_image = cv2.imread(input_image_path)
        if input_image is None:
            raise ValueError(f"Could not load image from {input_image_path}")

        # Preprocess the image
        preprocessed_image = self.preprocess_image(input_image)
        
        # Run inference
        binary_seg = self.predict(preprocessed_image)
        
        # Resize to match original size or target size for visualization
        binary_seg = cv2.resize(binary_seg, (512, 256))
        
        # Convert to RGB format for Pygame
        binary_seg = cv2.cvtColor(binary_seg, cv2.COLOR_GRAY2RGB)

        # Optional: Save the result
        if save_dir:
            save_path = os.path.join(save_dir, f"lane_{int(time.time())}.png")
            cv2.imwrite(save_path, binary_seg)

        # Convert NumPy array to Pygame surface
        surface = pygame.surfarray.make_surface(binary_seg.swapaxes(0, 1))  # Swap axes for correct orientation
        return surface

    def process_image(self, image):
        """Convert CARLA image to OpenCV format and process it with lane detection."""
        try:
            # Convert CARLA image to OpenCV format
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # BGRA format
            array = array[:, :, :3]  # Remove alpha channel

            # Save image with no compression loss
            filename = "carla_camera.png"
            cv2.imwrite(filename, array, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Set compression to 0

            # Return the filename for further processing
            return filename

        except Exception as e:
            print(f"Error processing image: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize pygame for visualization
    # pygame.init()
    # screen = pygame.display.set_mode((512, 256))
    # pygame.display.set_caption("Lane Detection")
    
    # Initialize the lane detector
    detector = KerasLaneDetector('output/lane_detector_final.keras')
    
    # Process a test image
    test_image_path = "/home/michel/Downloads/carla_camera.png"  # Replace with your test image path
    
    if os.path.exists(test_image_path):
        # Process and visualize
        lane_surface = detector.process_and_visualize(test_image_path)
        
        # Display the result
        screen.blit(lane_surface, (0, 0))
        pygame.display.flip()
        
        # Wait for user to close the window
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
        pygame.quit()
    else:
        print(f"Test image not found: {test_image_path}")