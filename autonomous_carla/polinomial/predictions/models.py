#!/usr/bin/env python3
# models.py - Model architecture and custom layers for lane detection

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Layer
from tensorflow.keras.models import Model

# Define custom layers for preprocessing
@tf.keras.utils.register_keras_serializable(package="LaneDetection")
class GaussianBlur(Layer):
    def __init__(self, **kwargs):
        super(GaussianBlur, self).__init__(**kwargs)
        # Define Gaussian kernel
        gaussian_kernel = tf.constant([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=tf.float32) / 16.0
        self.kernel = tf.reshape(gaussian_kernel, [3, 3, 1, 1])
        
    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    def get_config(self):
        config = super(GaussianBlur, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="LaneDetection")
class SobelFilter(Layer):
    def __init__(self, direction='x', **kwargs):
        super(SobelFilter, self).__init__(**kwargs)
        self.direction = direction
        # Define Sobel kernels
        if direction == 'x':
            sobel_kernel = tf.constant([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=tf.float32)
        else:  # y direction
            sobel_kernel = tf.constant([
                [-1, -2, -1],
                [0,  0,  0],
                [1,  2,  1]
            ], dtype=tf.float32)
        self.kernel = tf.reshape(sobel_kernel, [3, 3, 1, 1])
        
    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    def get_config(self):
        config = super(SobelFilter, self).get_config()
        config.update({"direction": self.direction})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="LaneDetection")
class GradientMagnitude(Layer):
    def __init__(self, **kwargs):
        super(GradientMagnitude, self).__init__(**kwargs)
        
    def call(self, inputs):
        grad_x, grad_y = inputs
        return tf.sqrt(tf.square(grad_x) + tf.square(grad_y))
    
    def get_config(self):
        config = super(GradientMagnitude, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="LaneDetection")
class AdaptiveThreshold(Layer):
    def __init__(self, k=30.0, **kwargs):
        super(AdaptiveThreshold, self).__init__(**kwargs)
        self.k = k
        
    def call(self, inputs):
        baseline = tf.reduce_mean(inputs)
        return tf.sigmoid(self.k * (inputs - baseline))
    
    def get_config(self):
        config = super(AdaptiveThreshold, self).get_config()
        config.update({"k": self.k})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_lane_detector(input_shape=(256, 256, 1)):
    """Build an enhanced U-Net style model for lane detection with integrated preprocessing."""
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, UpSampling2D, Input, Layer, 
        Concatenate, Dropout, BatchNormalization, SpatialDropout2D
    )
    from tensorflow.keras.models import Model
    
    # Main model inputs (raw images)
    inputs = Input(shape=input_shape, name="model_input")
    
    # Apply preprocessing layers
    x = GaussianBlur(name="gaussian_blur")(inputs)
    grad_x = SobelFilter(direction='x', name="sobel_x")(x)
    grad_y = SobelFilter(direction='y', name="sobel_y")(x)
    gradient = GradientMagnitude(name="gradient_magnitude")([grad_x, grad_y])
    x = GaussianBlur(name="non_max_suppression")(gradient)
    x = AdaptiveThreshold(name="adaptive_threshold")(x)
    
    # Enhanced Encoder with skip connections
    # Block 1
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same", name="encoder_conv1_1")(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same", name="encoder_conv1_2")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2), name="encoder_pool1")(conv1)
    pool1 = SpatialDropout2D(0.1)(pool1)
    
    # Block 2
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="encoder_conv2_1")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="encoder_conv2_2")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2), name="encoder_pool2")(conv2)
    pool2 = SpatialDropout2D(0.1)(pool2)
    
    # Block 3 (additional depth)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same", name="encoder_conv3_1")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same", name="encoder_conv3_2")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2), name="encoder_pool3")(conv3)
    pool3 = SpatialDropout2D(0.2)(pool3)
    
    # Middle (bottleneck)
    middle = Conv2D(256, (3, 3), activation="relu", padding="same", name="middle_conv1")(pool3)
    middle = BatchNormalization()(middle)
    middle = Conv2D(256, (3, 3), activation="relu", padding="same", name="middle_conv2")(middle)
    middle = BatchNormalization()(middle)
    middle = SpatialDropout2D(0.2)(middle)
    
    # Enhanced Decoder with skip connections
    # Block 3
    up3 = UpSampling2D((2, 2), name="decoder_up3")(middle)
    up3 = Conv2D(128, (2, 2), activation="relu", padding="same")(up3)
    concat3 = Concatenate()([up3, conv3])  # Skip connection
    deconv3 = Conv2D(128, (3, 3), activation="relu", padding="same", name="decoder_conv3_1")(concat3)
    deconv3 = BatchNormalization()(deconv3)
    deconv3 = Conv2D(128, (3, 3), activation="relu", padding="same", name="decoder_conv3_2")(deconv3)
    deconv3 = BatchNormalization()(deconv3)
    
    # Block 2
    up2 = UpSampling2D((2, 2), name="decoder_up2")(deconv3)
    up2 = Conv2D(64, (2, 2), activation="relu", padding="same")(up2)
    concat2 = Concatenate()([up2, conv2])  # Skip connection
    deconv2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="decoder_conv2_1")(concat2)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="decoder_conv2_2")(deconv2)
    deconv2 = BatchNormalization()(deconv2)
    
    # Block 1
    up1 = UpSampling2D((2, 2), name="decoder_up1")(deconv2)
    up1 = Conv2D(32, (2, 2), activation="relu", padding="same")(up1)
    concat1 = Concatenate()([up1, conv1])  # Skip connection
    deconv1 = Conv2D(32, (3, 3), activation="relu", padding="same", name="decoder_conv1_1")(concat1)
    deconv1 = BatchNormalization()(deconv1)
    deconv1 = Conv2D(32, (3, 3), activation="relu", padding="same", name="decoder_conv1_2")(deconv1)
    deconv1 = BatchNormalization()(deconv1)
    
    # Output with attention to lane structure
    lane_attention = Conv2D(16, (1, 1), activation="relu", name="lane_attention")(deconv1)
    outputs = Conv2D(1, (1, 1), activation="sigmoid", padding="same", name="output")(lane_attention)
    
    # Create the model
    model = Model(inputs, outputs, name="enhanced_lane_detector")
    return model

# Define a function to get custom objects dictionary
def get_custom_objects():
    """Return a dictionary of custom objects for model loading."""
    from metrics import BinaryMeanIoU
    from losses import (
        WeightedBinaryCrossEntropy, 
        FocalLoss, 
        DiceLoss, 
        CombinedLoss, 
        TverskyLoss, 
        BoundaryLoss
    )
    
    return {
        'GaussianBlur': GaussianBlur,
        'SobelFilter': SobelFilter,
        'GradientMagnitude': GradientMagnitude,
        'AdaptiveThreshold': AdaptiveThreshold,
        'BinaryMeanIoU': BinaryMeanIoU,
        'WeightedBinaryCrossEntropy': WeightedBinaryCrossEntropy,
        'FocalLoss': FocalLoss,
        'DiceLoss': DiceLoss,
        'CombinedLoss': CombinedLoss,
        'TverskyLoss': TverskyLoss,
        'BoundaryLoss': BoundaryLoss
    }