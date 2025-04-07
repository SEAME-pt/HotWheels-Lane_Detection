#!/usr/bin/env python3
# losses.py - Custom loss functions for lane detection

import tensorflow as tf
import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable(package="LaneDetection")
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Weighted binary cross-entropy loss that puts more emphasis on lane pixels.
    
    Args:
        pos_weight: Weight for positive class (lane pixels)
        neg_weight: Weight for negative class (non-lane pixels)
        from_logits: Whether model outputs logits or probabilities
        reduction: Type of tf.keras.losses.Reduction to apply
        name: Name of the loss function
    """
    def __init__(self, 
                 pos_weight=10.0, 
                 neg_weight=1.0, 
                 from_logits=False,
                 reduction="sum_over_batch_size",
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
            
        # Clip to prevent numerical instability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate weighted loss
        loss = - (self.pos_weight * y_true * K.log(y_pred) + 
                 self.neg_weight * (1 - y_true) * K.log(1 - y_pred))
        
        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "pos_weight": self.pos_weight,
            "neg_weight": self.neg_weight,
            "from_logits": self.from_logits
        })
        return config

@tf.keras.utils.register_keras_serializable(package="LaneDetection")
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal loss for dealing with class imbalance in lane detection.
    Puts more focus on hard-to-classify examples.
    
    Args:
        alpha: Weighting factor for class imbalance
        gamma: Focusing parameter
        from_logits: Whether model outputs logits or probabilities
        reduction: Type of tf.keras.losses.Reduction to apply
        name: Name of the loss function
    """
    def __init__(self, 
                 alpha=0.25, 
                 gamma=2.0, 
                 from_logits=False,
                 reduction="sum_over_batch_size",
                 name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
            
        # Clip to prevent numerical instability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # Calculate binary cross-entropy
        bce = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        
        # Apply modulating factor
        loss = alpha_factor * modulating_factor * bce
        
        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "from_logits": self.from_logits
        })
        return config

@tf.keras.utils.register_keras_serializable(package="LaneDetection")
class DiceLoss(tf.keras.losses.Loss):
    """
    Dice loss for lane detection, optimizing for overlap between prediction and ground truth.
    
    Args:
        smooth: Smoothing factor to avoid division by zero
        square: Whether to square the values before summing
        reduction: Type of tf.keras.losses.Reduction to apply
        name: Name of the loss function
    """
    def __init__(self,
                 smooth=1.0,
                 square=True,
                 reduction="sum_over_batch_size",
                 name='dice_loss'):
        super().__init__(reduction=reduction, name=name)
        self.smooth = smooth
        self.square = square
        
    def call(self, y_true, y_pred):
        # Flatten the predictions and targets
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        if self.square:
            intersection = K.sum(y_true * y_pred)
            union = K.sum(y_true**2) + K.sum(y_pred**2)
        else:
            intersection = K.sum(y_true * y_pred)
            union = K.sum(y_true) + K.sum(y_pred)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "smooth": self.smooth,
            "square": self.square
        })
        return config

@tf.keras.utils.register_keras_serializable(package="LaneDetection")
class CombinedLoss(tf.keras.losses.Loss):
    """
    Combined loss function using both binary cross-entropy and Dice loss.
    
    Args:
        bce_weight: Weight for binary cross-entropy component
        dice_weight: Weight for Dice loss component
        reduction: Type of tf.keras.losses.Reduction to apply
        name: Name of the loss function
    """
    def __init__(self,
                 bce_weight=0.5,
                 dice_weight=0.5,
                 reduction="sum_over_batch_size",
                 name='combined_loss'):
        super().__init__(reduction=reduction, name=name)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.dice = DiceLoss()
        
    def call(self, y_true, y_pred):
        bce_loss = self.bce(y_true, y_pred)
        dice_loss = self.dice(y_true, y_pred)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight
        })
        return config

@tf.keras.utils.register_keras_serializable(package="LaneDetection")
class TverskyLoss(tf.keras.losses.Loss):
    """
    Tversky loss, a generalization of Dice loss that allows different weights for
    false positives and false negatives.
    
    Args:
        alpha: Weight for false positives
        beta: Weight for false negatives
        smooth: Smoothing factor to avoid division by zero
        reduction: Type of tf.keras.losses.Reduction to apply
        name: Name of the loss function
    """
    def __init__(self,
                 alpha=0.3,
                 beta=0.7,
                 smooth=1.0,
                 reduction="sum_over_batch_size",
                 name='tversky_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def call(self, y_true, y_pred):
        # Flatten the predictions and targets
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        # Calculate true positives, false positives, and false negatives
        true_pos = K.sum(y_true * y_pred)
        false_pos = K.sum((1 - y_true) * y_pred)
        false_neg = K.sum(y_true * (1 - y_pred))
        
        # Calculate Tversky index
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        
        return 1 - tversky
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "smooth": self.smooth
        })
        return config

@tf.keras.utils.register_keras_serializable(package="LaneDetection") 
class BoundaryLoss(tf.keras.losses.Loss):
    """
    Boundary loss that focuses on the edges of lane markings.
    
    Args:
        theta: Thickness of the boundary
        reduction: Type of tf.keras.losses.Reduction to apply
        name: Name of the loss function
    """
    def __init__(self,
                 theta=3,
                 reduction="sum_over_batch_size",
                 name='boundary_loss'):
        super().__init__(reduction=reduction, name=name)
        self.theta = theta
        
    def call(self, y_true, y_pred):
        # Create boundary maps
        y_true_boundaries = self._create_boundaries(y_true)
        
        # Calculate weighted BCE focusing on boundaries
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weighted_bce = y_true_boundaries * bce
        
        return K.mean(weighted_bce)
    
    def _create_boundaries(self, mask):
        """Create a boundary map from a binary mask using dilation and erosion."""
        # Implementation using tf operations for dilation and erosion
        kernel_size = 2 * self.theta + 1
        
        # Create a pooling kernel for dilation
        dilated = tf.nn.max_pool2d(mask, ksize=kernel_size, strides=1, padding='SAME')
        
        # Create a pooling kernel for erosion
        eroded = -tf.nn.max_pool2d(-mask, ksize=kernel_size, strides=1, padding='SAME')
        
        # Boundary is the difference between dilation and erosion
        boundary = tf.cast(dilated - eroded > 0, tf.float32)
        
        return boundary
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "theta": self.theta
        })
        return config

# Dictionary of all available loss functions
def get_loss_functions():
    """Return a dictionary of all available custom loss functions."""
    return {
        'weighted_binary_crossentropy': WeightedBinaryCrossEntropy,
        'focal_loss': FocalLoss,
        'dice_loss': DiceLoss,
        'combined_loss': CombinedLoss,
        'tversky_loss': TverskyLoss,
        'boundary_loss': BoundaryLoss
    }

# Function to get a specific loss by name
def get_loss(loss_name, **kwargs):
    """
    Get a specific loss function by name with optional parameters.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss function
        
    Returns:
        A loss function instance
    """
    loss_functions = get_loss_functions()
    
    if loss_name in loss_functions:
        return loss_functions[loss_name](**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. Available options: {list(loss_functions.keys())}")