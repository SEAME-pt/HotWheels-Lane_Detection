#!/usr/bin/env python3
# visualize_layers.py - Visualize outputs from each layer of the lane detection model

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
import cv2

# Import configuration parameters
from config import *

# Import utility functions
from utils import set_seed, configure_gpu, create_visualization_dataset
from models import get_custom_objects

def create_intermediate_models(model):
    """
    Create separate models for each layer in the original model.
    
    Args:
        model: The original model
        
    Returns:
        list: List of (layer_name, intermediate_model) tuples
    """
    intermediate_models = []
    
    # For each layer in the model
    for i, layer in enumerate(model.layers):
        # Skip the input layer
        if i == 0:
            continue
            
        # Create an intermediate model that outputs this layer's activations
        intermediate_model = Model(
            inputs=model.input,
            outputs=layer.output,
            name=f"intermediate_{layer.name}"
        )
        
        intermediate_models.append((layer.name, intermediate_model))
        
    return intermediate_models

def visualize_layer_outputs(model_path, output_dir=None, sample_index=0):
    """
    Visualize the output of each layer in the model for a single input image.
    
    Args:
        model_path: Path to the saved model
        output_dir: Directory to save visualizations
        sample_index: Index of the sample to use for visualization
    """
    # Configure output directory
    output_dir = output_dir or os.path.join(OUTPUT_DIR, "layer_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure GPU
    configure_gpu()
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    print(f"🔍 Loading model from: {model_path}")
    
    # Load the model with custom objects
    custom_objects = get_custom_objects()
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get a sample image
    print("📷 Loading sample image...")
    vis_images, vis_masks = create_visualization_dataset(
        VIS_IMAGE_PATH, IMAGE_SIZE, batch_size=sample_index + 1
    )
    
    # Use the specified sample
    sample_image = vis_images[sample_index:sample_index+1]
    sample_mask = vis_masks[sample_index:sample_index+1]
    
    # Create intermediate models for each layer
    print("🔧 Creating intermediate models...")
    intermediate_models = create_intermediate_models(model)
    
    # Calculate grid size for the visualization
    grid_size = int(np.ceil(np.sqrt(len(intermediate_models) + 2)))  # +2 for input and ground truth
    
    # Create a large figure for all visualizations
    plt.figure(figsize=(grid_size * 4, grid_size * 4))
    
    # Plot the input image
    plt.subplot(grid_size, grid_size, 1)
    plt.title("Input Image")
    plt.imshow(sample_image[0].numpy().squeeze(), cmap='gray')
    plt.axis('off')
    
    # Plot the ground truth mask
    plt.subplot(grid_size, grid_size, 2)
    plt.title("Ground Truth")
    plt.imshow(sample_mask[0].numpy().squeeze(), cmap='gray')
    plt.axis('off')
    
    # Visualize the output of each layer
    print("🖼️ Generating layer visualizations...")
    
    for i, (layer_name, intermediate_model) in enumerate(intermediate_models):
        # Get the layer output
        layer_output = intermediate_model.predict(sample_image)
        
        # Create a separate detailed figure for this layer
        plt.figure(figsize=(12, 10))
        
        # Handle different output shapes
        if len(layer_output.shape) == 4:  # [batch, height, width, channels]
            if layer_output.shape[-1] == 1:
                # Single channel output
                plt.subplot(1, 1, 1)
                plt.title(f"Layer: {layer_name}")
                plt.imshow(layer_output[0].squeeze(), cmap='viridis')
                plt.colorbar(label='Activation')
                plt.axis('off')
            else:
                # Multiple channels - show a subset if there are many
                channels_to_show = min(16, layer_output.shape[-1])
                rows = int(np.ceil(np.sqrt(channels_to_show)))
                cols = int(np.ceil(channels_to_show / rows))
                
                for j in range(channels_to_show):
                    plt.subplot(rows, cols, j + 1)
                    plt.title(f"Channel {j}")
                    plt.imshow(layer_output[0, :, :, j], cmap='viridis')
                    plt.axis('off')
                
                plt.suptitle(f"Layer: {layer_name} - Selected Channels")
        else:
            # Handle other output shapes (like flattened layers)
            plt.title(f"Layer: {layer_name} - Output Shape: {layer_output.shape}")
            plt.text(0.5, 0.5, f"Non-spatial output\nShape: {layer_output.shape}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        # Save detailed layer visualization
        plt.tight_layout()
        layer_filename = os.path.join(output_dir, f"{i:03d}_{layer_name.replace('/', '_')}.png")
        plt.savefig(layer_filename)
        plt.close()
        
        # Add to the overview plot
        plt.figure(1)  # Switch back to main figure
        plt.subplot(grid_size, grid_size, i + 3)  # +3 because we already used 1 and 2
        
        if len(layer_output.shape) == 4:
            if layer_output.shape[-1] == 1:
                plt.imshow(layer_output[0].squeeze(), cmap='viridis')
            else:
                # For multiple channels, just show the first one in the overview
                plt.imshow(layer_output[0, :, :, 0], cmap='viridis')
        
        plt.title(layer_name, fontsize=8)
        plt.axis('off')
    
    # Save the overview plot
    plt.tight_layout()
    overview_filename = os.path.join(output_dir, "all_layers_overview.png")
    plt.savefig(overview_filename)
    plt.close()
    
    print(f"✅ Layer visualizations saved to: {output_dir}")
    
    # Create HTML visualization for easy browsing
    create_html_visualization(intermediate_models, output_dir)
    
    return output_dir

def create_html_visualization(intermediate_models, output_dir):
    """
    Create an HTML file to easily browse all layer visualizations.
    
    Args:
        intermediate_models: List of (layer_name, model) tuples
        output_dir: Directory containing the visualizations
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lane Detection Model Layer Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .layer-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
            .layer-card { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .layer-card h3 { padding: 10px; margin: 0; background-color: #f5f5f5; }
            .layer-card img { width: 100%; height: auto; display: block; }
            .overview { margin-bottom: 30px; }
            .overview img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h1>Lane Detection Model Layer Visualization</h1>
        
        <div class="overview">
            <h2>Overview of All Layers</h2>
            <img src="all_layers_overview.png" alt="Overview of all layers">
        </div>
        
        <h2>Individual Layer Outputs</h2>
        <div class="layer-grid">
    """
    
    # Add input and ground truth
    html_content += """
            <div class="layer-card">
                <h3>Input Image</h3>
                <img src="all_layers_overview.png" alt="Input Image">
            </div>
            <div class="layer-card">
                <h3>Ground Truth</h3>
                <img src="all_layers_overview.png" alt="Ground Truth">
            </div>
    """
    
    # Add each layer
    for i, (layer_name, _) in enumerate(intermediate_models):
        safe_name = layer_name.replace('/', '_')
        html_content += f"""
            <div class="layer-card">
                <h3>{layer_name}</h3>
                <img src="{i:03d}_{safe_name}.png" alt="{layer_name}">
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    html_path = os.path.join(output_dir, "visualization.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML visualization created: {html_path}")

def create_activation_flow(model_path, output_dir=None, sample_index=0):
    """
    Create a step-by-step visualization of the core model flow for documentation.
    This focuses on the key transformation steps rather than every layer.
    
    Args:
        model_path: Path to the saved model
        output_dir: Directory to save visualizations
        sample_index: Index of the sample to use for visualization
    """
    # Configure output directory
    output_dir = output_dir or os.path.join(OUTPUT_DIR, "activation_flow")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure GPU
    configure_gpu()
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Load the model with custom objects
    custom_objects = get_custom_objects()
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get a sample image
    vis_images, vis_masks = create_visualization_dataset(
        VIS_IMAGE_PATH, IMAGE_SIZE, batch_size=sample_index + 1
    )
    
    # Use the specified sample
    sample_image = vis_images[sample_index:sample_index+1]
    sample_mask = vis_masks[sample_index:sample_index+1]
    
    # Define key stages of the model to visualize
    key_layers = [
        "model_input",  # Input image
        "gaussian_blur",  # Blurred image
        "sobel_x",  # X gradient
        "sobel_y",  # Y gradient
        "gradient_magnitude",  # Combined gradient
        "adaptive_threshold",  # Thresholded gradient
        "encoder_conv1_2",  # First encoder block
        "encoder_conv2_2",  # Second encoder block
        "encoder_conv3_2",  # Third encoder block
        "middle_conv2",  # Bottleneck
        "decoder_conv3_2",  # First decoder block
        "decoder_conv2_2",  # Second decoder block
        "decoder_conv1_2",  # Third decoder block
        "lane_attention",  # Attention mechanism
        "output"  # Final output
    ]
    
    # Create intermediate models for the key layers
    key_models = []
    for layer_name in key_layers:
        # Special case for input
        if layer_name == "model_input":
            key_models.append((layer_name, None))
            continue
            
        # Find the layer by name
        try:
            layer = model.get_layer(layer_name)
            intermediate_model = Model(
                inputs=model.input,
                outputs=layer.output,
                name=f"intermediate_{layer_name}"
            )
            key_models.append((layer_name, intermediate_model))
        except ValueError:
            print(f"⚠️ Layer '{layer_name}' not found in model, skipping")
    
    # Create a figure for the sequential flow
    plt.figure(figsize=(15, len(key_models) * 4))
    
    # Process and visualize each stage
    flow_images = []
    
    for i, (layer_name, intermediate_model) in enumerate(key_models):
        plt.subplot(len(key_models), 1, i + 1)
        
        if layer_name == "model_input":
            # For input, show the original image
            image_to_display = sample_image[0].numpy().squeeze()
            plt.title("Stage 1: Input Image")
            plt.imshow(image_to_display, cmap='gray')
            flow_images.append(("Input Image", image_to_display))
        else:
            # Get intermediate output
            layer_output = intermediate_model.predict(sample_image)
            
            # Format for display
            if len(layer_output.shape) == 4:
                if layer_output.shape[-1] == 1:
                    # Single channel output
                    image_to_display = layer_output[0].squeeze()
                else:
                    # For multi-channel outputs, create a visualization that combines channels
                    # We'll use the first channel for the flow visualization
                    image_to_display = layer_output[0, :, :, 0]
            else:
                # Skip non-image outputs
                continue
                
            plt.title(f"Stage {i+1}: {layer_name}")
            
            # Choose colormap based on layer type
            if "sobel" in layer_name or "gradient" in layer_name:
                cmap = 'coolwarm'
            elif "threshold" in layer_name or "output" in layer_name:
                cmap = 'gray'
            else:
                cmap = 'viridis'
                
            plt.imshow(image_to_display, cmap=cmap)
            flow_images.append((layer_name, image_to_display))
        
        plt.axis('off')
    
    plt.tight_layout()
    flow_filename = os.path.join(output_dir, "model_activation_flow.png")
    plt.savefig(flow_filename)
    plt.close()
    
    # Create a more design-friendly visualization with arrows
    create_flow_diagram(flow_images, output_dir)
    
    print(f"✅ Model activation flow saved to: {output_dir}")
    return output_dir

def create_flow_diagram(flow_images, output_dir):
    """
    Create a more visually appealing flow diagram with arrows.
    
    Args:
        flow_images: List of (layer_name, image) tuples
        output_dir: Directory to save visualizations
    """
    # Calculate layout
    n_images = len(flow_images)
    fig_height = max(15, n_images * 2)
    
    fig, axes = plt.subplots(n_images, 2, figsize=(12, fig_height), 
                             gridspec_kw={'width_ratios': [3, 1]})
    
    # If only one image, make axes a 2D array
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    # Create arrow properties
    arrow_props = dict(arrowstyle="->", color='blue', lw=2)
    
    for i, (layer_name, image) in enumerate(flow_images):
        # Display image
        ax_img = axes[i, 0]
        
        # Normalize image for display
        if image.min() != image.max():
            display_img = (image - image.min()) / (image.max() - image.min())
        else:
            display_img = image
            
        # Choose colormap
        if "sobel" in layer_name.lower() or "gradient" in layer_name.lower():
            cmap = 'coolwarm'
        elif "threshold" in layer_name.lower() or "output" in layer_name.lower() or "input" in layer_name.lower():
            cmap = 'gray'
        else:
            cmap = 'viridis'
            
        im = ax_img.imshow(display_img, cmap=cmap)
        ax_img.set_title(f"Stage {i+1}: {layer_name}")
        ax_img.axis('off')
        
        # Add colorbar
        cbar_ax = axes[i, 1]
        fig.colorbar(im, cax=cbar_ax)
        
        # Add arrow to next image
        if i < n_images - 1:
            ax_next = axes[i+1, 0]
            fig.canvas.draw()  # Update the figure
            
            # Get axis positions
            bbox1 = ax_img.get_position()
            bbox2 = ax_next.get_position()
            
            # Create arrow annotation
            arrow_x = 0.5 * (bbox1.x0 + bbox1.x1)
            fig.patches.extend([
                plt.Arrow(arrow_x, bbox1.y0, 0, bbox2.y1 - bbox1.y0, 
                         width=0.03, color='blue')
            ])
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    flow_diagram_path = os.path.join(output_dir, "model_flow_diagram.png")
    plt.savefig(flow_diagram_path)
    plt.close()
    
    # Also create individual stage images for documentation
    for i, (layer_name, image) in enumerate(flow_images):
        plt.figure(figsize=(8, 8))
        
        # Normalize image for display
        if image.min() != image.max():
            display_img = (image - image.min()) / (image.max() - image.min())
        else:
            display_img = image
            
        # Choose colormap
        if "sobel" in layer_name.lower() or "gradient" in layer_name.lower():
            cmap = 'coolwarm'
        elif "threshold" in layer_name.lower() or "output" in layer_name.lower() or "input" in layer_name.lower():
            cmap = 'gray'
        else:
            cmap = 'viridis'
            
        plt.imshow(display_img, cmap=cmap)
        plt.colorbar(label='Activation')
        plt.title(f"Stage {i+1}: {layer_name}")
        plt.axis('off')
        
        stage_path = os.path.join(output_dir, f"stage_{i+1:02d}_{layer_name.replace('/', '_')}.png")
        plt.savefig(stage_path)
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize layer outputs of a lane detection model')
    parser.add_argument('--model_path', type=str, default=os.path.join(OUTPUT_DIR, "lane_detector_final.keras"),
                       help='Path to the saved model')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualizations')
    parser.add_argument('--sample_index', type=int, default=0,
                       help='Index of the sample to use for visualization')
    parser.add_argument('--mode', type=str, choices=['detailed', 'flow', 'both'], default='both',
                       help='Visualization mode: detailed (all layers), flow (key stages), or both')
    
    args = parser.parse_args()
    
    # Run the visualization
    if args.mode in ['detailed', 'both']:
        visualize_layer_outputs(
            model_path=args.model_path,
            output_dir=args.output_dir,
            sample_index=args.sample_index
        )
    
    if args.mode in ['flow', 'both']:
        create_activation_flow(
            model_path=args.model_path,
            output_dir=args.output_dir,
            sample_index=args.sample_index
        )