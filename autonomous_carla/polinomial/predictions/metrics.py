#!/usr/bin/env python3
# metrics.py - Custom metrics for lane detection

import tensorflow as tf

# Custom IoU metric for binary segmentation
@tf.keras.utils.register_keras_serializable(package="LaneDetection")
class BinaryMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='binary_mean_iou', **kwargs):
        super(BinaryMeanIoU, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Apply threshold to predictions
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
        
        # Calculate IoU for each sample in the batch
        iou = tf.where(union > 0, intersection / union, tf.ones_like(intersection))
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        
        # Update state variables
        self.iou_sum.assign_add(tf.reduce_sum(iou))
        self.total_samples.assign_add(batch_size)
        
    def result(self):
        return self.iou_sum / self.total_samples
        
    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.total_samples.assign(0.0)
        
    def get_config(self):
        config = super(BinaryMeanIoU, self).get_config()
        config.update({"threshold": self.threshold})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)