#!/usr/bin/env python3
# train.py - Main training script for lane detection

import os
import tensorflow as tf

# Import our modules
from models import build_lane_detector, get_custom_objects
from metrics import BinaryMeanIoU
from utils import (
    set_seed, 
    configure_gpu, 
    create_visualization_dataset, 
    visualize_samples, 
    create_callbacks, 
    plot_history
)
from losses import get_loss

# Import the dataset loader
from lane_dataset import create_lane_datasets

# Import configuration parameters
from config import *

def main():
    """Main training function."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Configure GPU memory growth
    configure_gpu()
    
    print("🔄 Loading lane detection datasets...")
    train_dataset, val_dataset, total_samples = create_lane_datasets(
        base_dir=DATASET_PATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=SEED
    )
    
    # Calculate steps per epoch
    train_samples = int(total_samples * (1 - VALIDATION_SPLIT))
    val_samples = total_samples - train_samples
    
    steps_per_epoch = train_samples // BATCH_SIZE
    validation_steps = val_samples // BATCH_SIZE
    
    print(f"Total samples: {total_samples}")
    print(f"Training steps per epoch: {steps_per_epoch}")
    print(f"Validation steps per epoch: {validation_steps}")
    
    # Create and compile the model
    model = build_lane_detector(input_shape=IMAGE_SIZE + (1,))
    
    # Compile with custom IoU metric
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=get_loss('combined_loss', bce_weight=0.3, dice_weight=0.7),  # Try the combined loss
    metrics=["accuracy", BinaryMeanIoU(threshold=0.5)]
)
    
    # Display model summary
    model.summary()
    
    # Get visualization data
    vis_images, vis_masks = create_visualization_dataset(DATASET_PATH, IMAGE_SIZE, batch_size=3)
    
    # Visualize the initial samples
    visualize_samples(
        vis_images, 
        vis_masks, 
        filename=os.path.join(OUTPUT_DIR, 'dataset_visualization.png')
    )
    
    # Train the model
    print("🚀 Starting model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=create_callbacks(
            checkpoints_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
            logs_dir=os.path.join(OUTPUT_DIR, "logs")
        ),
        verbose=1
    )
    
    # Save the final model with proper serialization support
    try:
        # Try to save in TF SavedModel format (more reliable)
        model.save(os.path.join(OUTPUT_DIR, "lane_detector_final"), save_format='tf')
        print(f"✅ Model saved in TensorFlow SavedModel format as '{os.path.join(OUTPUT_DIR, 'lane_detector_final')}'")
    except Exception as e:
        print(f"Warning: Could not save in TF format: {e}")
        # Fall back to Keras H5 format
        model.save(os.path.join(OUTPUT_DIR, "lane_detector_final.keras"))
        print(f"✅ Model saved in Keras format as '{os.path.join(OUTPUT_DIR, 'lane_detector_final.keras')}'")
    
    # Visualize with predictions after training
    visualize_samples(
        vis_images, 
        vis_masks, 
        model,
        filename=os.path.join(OUTPUT_DIR, 'prediction_visualization.png')
    )
    
    # Plot training history
    plot_history(
        history, 
        filename=os.path.join(OUTPUT_DIR, 'training_history.png')
    )
    
    print("✅ All tasks completed successfully!")

if __name__ == "__main__":
    main()