import cv2
import numpy as np
import os
import torch
import pygame
import time
from lane_detector import ENet

# Load the pre-trained model
model_path =  'ENET.pth'
enet_model = ENet(2, 4)

# Load the trained model's weights
enet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
enet_model.eval()  # Set the model to evaluation mode

def sharpen_image(image):
    """Applies unsharp masking to sharpen the image."""

    # Convert to grayscale if needed
    blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def process_and_visualize(input_image_path, save_dir="lane_output"):
    """Processes the image through the lane detection model and displays it in Pygame."""

    os.makedirs(save_dir, exist_ok=True)

    # Load and preprocess the input image
    input_image = cv2.imread(input_image_path)
    input_image = cv2.resize(input_image, (512, 256))  # Resize to match model input
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = input_image[..., None]  # Add channel dimension
    input_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1).unsqueeze(0)  # Convert to tensor

    # Pass through the model
    with torch.no_grad():
        binary_logits, _ = enet_model(input_tensor)

    # Get lane segmentation mask
    binary_seg = torch.argmax(binary_logits, dim=1).squeeze().numpy()

    # Convert to RGB format for Pygame
    binary_seg = (binary_seg * 255).astype(np.uint8)  # Scale values to 0-255
    binary_seg = cv2.cvtColor(binary_seg, cv2.COLOR_GRAY2RGB)  # Convert to RGB format

    # Save the image to a file
    #save_path = os.path.join(save_dir, f"lane_{int(time.time())}.png")
    #cv2.imwrite(save_path, binary_seg)  # Save as PNG

    # Convert NumPy array to Pygame surface
    surface = pygame.surfarray.make_surface(binary_seg.swapaxes(0, 1))  # Swap axes for correct orientation
    return surface

def process_image(image):
    """Convert CARLA image to OpenCV format and process it with lane detection."""
    global captured_image  # Store the processed image for display

    try:
        # Convert CARLA image to OpenCV format
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA format
        array = array[:, :, :3]  # Remove alpha channel

        # Save image with no compression loss
        filename = "carla_camera.png"
        cv2.imwrite(filename, array, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Set compression to 0

        # Process image using ENet and store the surface for display
        return filename
        #captured_image = process_and_visualize(filename)

    except Exception as e:
        print(f"Error processing image: {e}")
