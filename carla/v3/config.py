#!/usr/bin/env python3
# config.py - Configuration parameters for lane detection

# =========================================
# Configuration Parameters
# =========================================

# Dataset configuration
DATASET_PATH = "/home/seame/dev/CULaneDataset/trainingAndValidation"
# DATASET_PATH = "/home/seame/dev/CULaneDataset/trainingAndValidation/driver_23_30frame/05151640_0419.MP4"
TESTING_DATASET_PATH = "/home/seame/dev/CULaneDataset/testing"
# TESTING_DATASET_PATH = "/home/seame/dev/CULaneDataset/testing/driver_100_30frame"
VIS_IMAGE_PATH = "/home/seame/dev/CULaneDataset/testing/driver_193_90frame/06042037_0520.MP4"
IMAGE_SIZE = (256, 256)
VALIDATION_SPLIT = 0.2
SEED = 31415

# Training configuration
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001

# Model configuration
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1

# Output directory
OUTPUT_DIR = "./output"

# Gaussian Blur parameters
GAUSSIAN_KERNEL_SIZE = 3
GAUSSIAN_SIGMA = 1.0

# Adaptive Threshold parameters
THRESHOLD_K = 30.0

# Callbacks configuration
EARLY_STOPPING_PATIENCE = 2
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6