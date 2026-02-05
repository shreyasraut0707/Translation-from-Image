# Configuration settings for the project

import os

# base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "IIIT5K-Word_V3.0", "IIIT5K")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# dataset file paths
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
TRAIN_MAT = os.path.join(DATASET_DIR, "traindata.mat")
TEST_MAT = os.path.join(DATASET_DIR, "testdata.mat")

# model file paths
MODEL_PATH = os.path.join(MODELS_DIR, "crnn_model.h5")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "best_weights.h5")

# create directories if they dont exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# model training settings
IMG_WIDTH = 128
IMG_HEIGHT = 32
MAX_TEXT_LENGTH = 20
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# character set for OCR (letters, numbers)
CHARACTERS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARACTERS) + 1  # extra class for blank token

# translation settings
TARGET_LANGUAGE = "hi"  # hindi
SOURCE_LANGUAGE = "en"  # english

# GUI window settings
WINDOW_TITLE = "Translation from Image"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
FONT_FAMILY = "Arial"
FONT_SIZE_TITLE = 16
FONT_SIZE_NORMAL = 12

# video processing settings
VIDEO_FRAME_INTERVAL = 30  # process every 30th frame
MAX_FRAMES_TO_PROCESS = 100
