#!/usr/bin/env python3
# utils.py - Utility functions for lane detection

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Set random seed for reproducibility
def set_seed(seed=31415):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Configure GPU memory growth
def configure_gpu():
    """Configure GPU memory growth to avoid memory allocation issues."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"GPU memory growth configuration error: {e}")

# Create a visualization dataset just for verification
def create_visualization_dataset(base_dir, image_size=(256, 256), batch_size=3):
    """Creates a small dataset just for visualization purposes"""
    # Find image paths
    image_paths = []
    label_paths = []
    
    for root, _, files in os.walk(base_dir):
        jpg_files = [f for f in files if f.endswith(".jpg")][:batch_size]  # Just take a few
        for jpg_file in jpg_files:
            img_path = os.path.join(root, jpg_file)
            lbl_path = img_path.replace(".jpg", ".lines.txt")
            if os.path.exists(lbl_path):
                image_paths.append(img_path)
                label_paths.append(lbl_path)
                if len(image_paths) >= batch_size:
                    break
        if len(image_paths) >= batch_size:
            break
    
    # Load and process images manually
    images = []
    masks = []
    
    for img_path, lbl_path in zip(image_paths, label_paths):
        # Load image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, image_size)
        
        # Get original dimensions
        orig_img = tf.io.decode_jpeg(tf.io.read_file(img_path))
        orig_h, orig_w = orig_img.shape[0], orig_img.shape[1]
        
        # Load mask
        mask = np.zeros(image_size, dtype=np.float32)
        try:
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    points = list(map(float, line.strip().split()))
                    if len(points) < 4:
                        continue
                    
                    # Scale points
                    scaled_points = []
                    for i in range(0, len(points) - 1, 2):
                        x = int(round((points[i] / orig_w) * image_size[1]))
                        y = int(round((points[i + 1] / orig_h) * image_size[0]))
                        x = min(max(x, 0), image_size[1] - 1)
                        y = min(max(y, 0), image_size[0] - 1)
                        scaled_points.append((x, y))
                    
                    # Draw lines
                    for i in range(len(scaled_points) - 1):
                        cv2.line(
                            mask, 
                            scaled_points[i], 
                            scaled_points[i + 1],
                            color=1.0, 
                            thickness=2
                        )
        except Exception as e:
            print(f"Error loading mask for visualization: {e}")
        
        mask = np.expand_dims(mask, axis=-1)
        
        images.append(img)
        masks.append(mask)
    
    # Convert to tensors
    images_tensor = tf.stack(images)
    masks_tensor = tf.convert_to_tensor(masks)
    
    return images_tensor, masks_tensor

# Visualize samples function
def visualize_samples(images, masks, model=None, filename='dataset_visualization.png'):
    """Visualize samples with optional model predictions"""
    n_samples = len(images)
    cols = 3 if model is None else 4
    
    plt.figure(figsize=(cols * 5, n_samples * 5))
    
    for j in range(n_samples):
        # Display image
        plt.subplot(n_samples, cols, j*cols + 1)
        plt.title(f"Image {j+1}")
        plt.imshow(images[j].numpy().squeeze(), cmap='gray')
        plt.axis('off')
        
        # Display mask
        plt.subplot(n_samples, cols, j*cols + 2)
        plt.title(f"Ground Truth Mask {j+1}")
        plt.imshow(masks[j].numpy().squeeze(), cmap='gray')
        plt.axis('off')
        
        # Display overlay
        plt.subplot(n_samples, cols, j*cols + 3)
        plt.title(f"Overlay {j+1}")
        plt.imshow(images[j].numpy().squeeze(), cmap='gray')
        plt.imshow(masks[j].numpy().squeeze(), cmap='hot', alpha=0.5)
        plt.axis('off')
        
        # If model is provided, show predictions
        if model is not None:
            pred = model.predict(np.expand_dims(images[j], axis=0))[0]
            plt.subplot(n_samples, cols, j*cols + 4)
            plt.title(f"Prediction {j+1}")
            plt.imshow(images[j].numpy().squeeze(), cmap='gray')
            plt.imshow(pred.squeeze(), cmap='spring', alpha=0.5)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Visualization saved to '{filename}'")

# Create callbacks for training
def create_callbacks(checkpoints_dir="./checkpoints", logs_dir="./logs"):
    """Create callbacks for training the model."""
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
    
    # Import configuration parameters
    from config import EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR
    
    # Create directories if they don't exist
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return [
        ModelCheckpoint(
            filepath=os.path.join(checkpoints_dir, "lane_detector_best.keras"),
            save_best_only=True,
            monitor="val_loss"
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        TensorBoard(
            log_dir=logs_dir,
            update_freq="epoch"
        )
    ]

# Plot training history
def plot_history(history, filename='training_history.png'):
    """Plot training and validation metrics history."""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot IoU
    plt.subplot(1, 3, 3)
    plt.plot(history.history['binary_mean_iou'], label='Training IoU')
    plt.plot(history.history['val_binary_mean_iou'], label='Validation IoU')
    plt.title('IoU History')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Training history saved to '{filename}'")