#!/usr/bin/env python3
# evaluate.py - Evaluation script for lane detection model

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Import configuration
from config import *

# Import our utility functions
from utils import (
    set_seed, 
    configure_gpu, 
    create_visualization_dataset, 
    visualize_samples
)

# Import custom objects for model loading
from models import get_custom_objects

def evaluate_model(model_path, dataset_path=None, output_dir=None, batch_size=None, n_samples=10):
    """
    Evaluate a trained lane detection model on test data.
    
    Args:
        model_path: Path to the saved model
        dataset_path: Path to the dataset (defaults to config.DATASET_PATH)
        output_dir: Directory to save results (defaults to config.OUTPUT_DIR)
        batch_size: Batch size for evaluation (defaults to config.BATCH_SIZE)
        n_samples: Number of samples to visualize
    """
    # Use default values from config if not provided
    dataset_path = dataset_path or TESTING_DATASET_PATH
    output_dir = output_dir or os.path.join(OUTPUT_DIR, "evaluation")
    batch_size = batch_size or BATCH_SIZE
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure GPU
    configure_gpu()
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    print(f"🔍 Evaluating model from: {model_path}")
    
    # Load the model with custom objects
    custom_objects = get_custom_objects()
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create visualization dataset
    print(f"📊 Creating visualization dataset with {n_samples} samples...")
    vis_images, vis_masks = create_visualization_dataset(
        dataset_path, IMAGE_SIZE, batch_size=n_samples
    )
    
    # Evaluate model on samples
    print("📝 Evaluating model...")
    loss, accuracy, iou = model.evaluate(vis_images, vis_masks, verbose=1)
    
    print(f"📊 Evaluation results:")
    print(f"  - Loss: {loss:.4f}")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - IoU: {iou:.4f}")
    
    # Visualize predictions
    print("🖼️ Generating visualizations...")
    visualize_samples(
        vis_images, 
        vis_masks, 
        model,
        filename=os.path.join(output_dir, 'evaluation_visualization.png')
    )
    
    # Generate detailed metrics for each sample
    print("📊 Computing individual sample metrics...")
    predictions = model.predict(vis_images)
    
    # Compute metrics for each sample
    sample_metrics = []
    for i in range(len(vis_images)):
        # Get true and predicted masks
        true_mask = vis_masks[i].numpy().squeeze()
        pred_mask = predictions[i].squeeze() > 0.5  # Apply threshold
        
        # Convert to binary
        true_binary = true_mask > 0
        pred_binary = pred_mask > 0
        
        # Calculate metrics
        intersection = np.logical_and(true_binary, pred_binary).sum()
        union = np.logical_or(true_binary, pred_binary).sum()
        iou = intersection / union if union > 0 else 0
        accuracy = np.mean(true_binary == pred_binary)
        
        # Calculate precision and recall
        true_positives = np.logical_and(true_binary, pred_binary).sum()
        false_positives = np.logical_and(np.logical_not(true_binary), pred_binary).sum()
        false_negatives = np.logical_and(true_binary, np.logical_not(pred_binary)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        sample_metrics.append({
            'sample': i+1,
            'iou': iou,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Save metrics to CSV
    import csv
    metrics_file = os.path.join(output_dir, 'sample_metrics.csv')
    with open(metrics_file, 'w', newline='') as csvfile:
        fieldnames = ['sample', 'iou', 'accuracy', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in sample_metrics:
            writer.writerow(metrics)
    
    print(f"✅ Sample metrics saved to: {metrics_file}")
    
    # Plot metrics
    metrics_fig = plt.figure(figsize=(15, 10))
    
    # IoU plot
    plt.subplot(2, 2, 1)
    plt.bar(range(1, len(sample_metrics) + 1), [m['iou'] for m in sample_metrics])
    plt.axhline(y=np.mean([m['iou'] for m in sample_metrics]), color='r', linestyle='-', label=f"Avg: {np.mean([m['iou'] for m in sample_metrics]):.3f}")
    plt.xlabel('Sample')
    plt.ylabel('IoU')
    plt.title('IoU by Sample')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.bar(range(1, len(sample_metrics) + 1), [m['accuracy'] for m in sample_metrics])
    plt.axhline(y=np.mean([m['accuracy'] for m in sample_metrics]), color='r', linestyle='-', label=f"Avg: {np.mean([m['accuracy'] for m in sample_metrics]):.3f}")
    plt.xlabel('Sample')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Sample')
    plt.legend()
    
    # Precision/Recall plot
    plt.subplot(2, 2, 3)
    x = range(1, len(sample_metrics) + 1)
    plt.bar([i - 0.2 for i in x], [m['precision'] for m in sample_metrics], width=0.4, label='Precision')
    plt.bar([i + 0.2 for i in x], [m['recall'] for m in sample_metrics], width=0.4, label='Recall')
    plt.xlabel('Sample')
    plt.ylabel('Score')
    plt.title('Precision & Recall by Sample')
    plt.legend()
    
    # F1 plot
    plt.subplot(2, 2, 4)
    plt.bar(range(1, len(sample_metrics) + 1), [m['f1'] for m in sample_metrics])
    plt.axhline(y=np.mean([m['f1'] for m in sample_metrics]), color='r', linestyle='-', label=f"Avg: {np.mean([m['f1'] for m in sample_metrics]):.3f}")
    plt.xlabel('Sample')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Sample')
    plt.legend()
    
    plt.tight_layout()
    metrics_plot_file = os.path.join(output_dir, 'sample_metrics.png')
    plt.savefig(metrics_plot_file)
    plt.close()
    
    print(f"✅ Metrics visualization saved to: {metrics_plot_file}")
    print("✅ Evaluation completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a lane detection model')
    parser.add_argument('--model_path', type=str, default=os.path.join(OUTPUT_DIR, "lane_detector_final.keras"),
                        help='Path to the saved model')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for evaluation')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        n_samples=args.n_samples
    )