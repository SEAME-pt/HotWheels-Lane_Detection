import os
import tensorflow as tf
import numpy as np
import cv2

class LaneDataset:
    """
    A dataset loader for lane detection that follows TensorFlow's best practices
    for efficient loading and preprocessing of data.
    """
    
    def __init__(self, 
                 base_dir,
                 image_size=(256, 256),
                 batch_size=32,
                 shuffle=True,
                 seed=31415,
                 validation_split=0.2,
                 line_width=2):
        """
        Initialize the lane detection dataset loader.
        
        Args:
            base_dir (str): Base directory containing images and label files
            image_size (tuple): Target size for images (height, width)
            batch_size (int): Batch size for training
            shuffle (bool): Whether to shuffle the dataset
            seed (int): Random seed for reproducibility
            validation_split (float): Fraction of data to use for validation
            line_width (int): Thickness of lane lines in mask generation
        """
        self.base_dir = base_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.validation_split = validation_split
        self.line_width = line_width
        
        # Set random seed for reproducibility
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # Find all valid image-label pairs
        self.image_paths, self.label_paths = self._find_data_pairs()
        print(f"Found {len(self.image_paths)} valid image-label pairs")
        
        # Shuffle consistently (same for train/val split)
        if shuffle:
            # Create index permutation
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            
            # Apply permutation
            self.image_paths = [self.image_paths[i] for i in indices]
            self.label_paths = [self.label_paths[i] for i in indices]
    
    def _find_data_pairs(self):
        """Find all valid image-label pairs recursively in the dataset directory."""
        image_paths = []
        label_paths = []
        
        # Walk through all subdirectories
        for root, _, files in os.walk(self.base_dir):
            # Filter for jpg files
            jpg_files = sorted([f for f in files if f.endswith(".jpg")])
            
            for jpg_file in jpg_files:
                img_path = os.path.join(root, jpg_file)
                lbl_path = img_path.replace(".jpg", ".lines.txt")
                
                # Only include if label file exists
                if os.path.exists(lbl_path):
                    # Check if image is valid
                    try:
                        with open(img_path, 'rb') as f:
                            image_data = f.read()
                        tf.io.decode_jpeg(image_data)  # Test decode
                        image_paths.append(img_path)
                        label_paths.append(lbl_path)
                    except Exception as e:
                        print(f"⚠️ Skipping invalid image {img_path}: {e}")
        
        return image_paths, label_paths
    
    def _parse_image_and_label(self, image_path, label_path):
        """Parse a single image and its corresponding label file."""
        # Read image
        image_data = tf.io.read_file(image_path)
        # Decode to tensor
        image = tf.image.decode_jpeg(image_data, channels=1)
        # Convert to float and normalize
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize image
        image = tf.image.resize(image, self.image_size)
        
        # Get original image dimensions for label scaling
        orig_img = tf.io.decode_jpeg(image_data)
        orig_height = tf.shape(orig_img)[0]
        orig_width = tf.shape(orig_img)[1]
        
        # Read and process label
        mask = tf.py_function(
            func=self._load_lane_mask,
            inp=[label_path, orig_height, orig_width],
            Tout=tf.float32
        )
        mask = tf.ensure_shape(mask, [*self.image_size, 1])
        
        return image, mask
    
    def _load_lane_mask(self, label_path, orig_height, orig_width):
        """
        Load lane annotations from label file and create a binary mask.
        
        Args:
            label_path: Path to the .lines.txt file
            orig_height: Original image height
            orig_width: Original image width
            
        Returns:
            Binary mask as float32 tensor with shape [height, width, 1]
        """
        label_path_str = label_path.numpy().decode("utf-8")
        orig_h, orig_w = orig_height.numpy(), orig_width.numpy()
        target_h, target_w = self.image_size
        
        # Create empty mask
        lane_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        
        try:
            with open(label_path_str, "r") as file:
                lines = file.readlines()
                
                for line in lines:
                    points = list(map(float, line.strip().split()))
                    if len(points) < 4:  # Need at least 2 points (x1,y1,x2,y2)
                        continue
                    
                    # Scale and draw points
                    scaled_points = []
                    for i in range(0, len(points) - 1, 2):
                        x = int(round((points[i] / orig_w) * target_w))
                        y = int(round((points[i + 1] / orig_h) * target_h))
                        x = min(max(x, 0), target_w - 1)
                        y = min(max(y, 0), target_h - 1)
                        scaled_points.append((x, y))
                    
                    # Draw lines between points
                    for i in range(len(scaled_points) - 1):
                        cv2.line(
                            lane_mask, 
                            scaled_points[i], 
                            scaled_points[i + 1],
                            color=255, 
                            thickness=self.line_width
                        )
                    
        except Exception as e:
            print(f"⚠️ Error loading lanes from {label_path_str}: {e}")
        
        # Normalize to [0, 1] and add channel dimension
        lane_mask = lane_mask.astype(np.float32) / 255.0
        lane_mask = np.expand_dims(lane_mask, axis=-1)
        
        return lane_mask
    
    def _create_dataset(self, image_paths, label_paths):
        """Create a TensorFlow dataset from lists of image and label paths."""
        # Create dataset of paths
        paths_ds = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        
        # Map to actual images and masks
        image_ds = paths_ds.map(
            self._parse_image_and_label,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Filter out any invalid samples
        def is_valid(image, mask):
            # Check for NaN values
            no_nans = tf.logical_and(
                tf.reduce_all(tf.math.is_finite(image)),
                tf.reduce_all(tf.math.is_finite(mask))
            )
            
            # Check for valid range (0-1)
            valid_range = tf.logical_and(
                tf.logical_and(
                    tf.reduce_all(tf.greater_equal(image, 0)),
                    tf.reduce_all(tf.less_equal(image, 1))
                ),
                tf.logical_and(
                    tf.reduce_all(tf.greater_equal(mask, 0)),
                    tf.reduce_all(tf.less_equal(mask, 1))
                )
            )
            
            return tf.logical_and(no_nans, valid_range)
        
        image_ds = image_ds.filter(is_valid)
        
        return image_ds
    
    def get_dataset(self, training=True):
        """
        Get the prepared dataset for training or validation.
        
        Args:
            training (bool): Whether to return the training set (True) or validation set (False)
            
        Returns:
            tf.data.Dataset: Prepared dataset ready for model training or evaluation
        """
        total_samples = len(self.image_paths)
        split_idx = int(total_samples * (1 - self.validation_split))
        
        if training:
            # Training set
            image_paths = self.image_paths[:split_idx]
            label_paths = self.label_paths[:split_idx]
            print(f"Creating training dataset with {len(image_paths)} samples")
        else:
            # Validation set
            image_paths = self.image_paths[split_idx:]
            label_paths = self.label_paths[split_idx:]
            print(f"Creating validation dataset with {len(image_paths)} samples")
        
        # Create the base dataset
        dataset = self._create_dataset(image_paths, label_paths)
        
        if training and self.shuffle:
            # For training, apply shuffle with a reasonable buffer size
            buffer_size = min(len(image_paths), 1000)  # Limit buffer to prevent OOM
            dataset = dataset.shuffle(buffer_size=buffer_size, seed=self.seed, reshuffle_each_iteration=True)
        
        # Batch the data first
        dataset = dataset.batch(self.batch_size, drop_remainder=training)
        
        # Important: For training, repeat AFTER batching to avoid the warning
        if training:
            dataset = dataset.repeat()
        
        # Prefetch for both training and validation
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def create_lane_datasets(base_dir, 
                         batch_size=32, 
                         image_size=(256, 256),
                         validation_split=0.2,
                         seed=31415):
    """
    Convenience function to create training and validation datasets.
    
    Args:
        base_dir (str): Base directory containing images and label files
        batch_size (int): Batch size for training
        image_size (tuple): Target size for images (height, width)
        validation_split (float): Fraction of data to use for validation
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (training_dataset, validation_dataset)
    """
    # Create dataset loader
    lane_data = LaneDataset(
        base_dir=base_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split
    )
    
    # Get training and validation datasets
    train_ds = lane_data.get_dataset(training=True)
    val_ds = lane_data.get_dataset(training=False)
    
    return train_ds, val_ds, len(lane_data.image_paths)

# Example Usage:
if __name__ == "__main__":
    # Example usage
    train_dataset, val_dataset, total_samples = create_lane_datasets(
        base_dir="/path/to/dataset", 
        batch_size=8
    )
    
    # Check dataset shapes
    for images, masks in train_dataset.take(1):
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")