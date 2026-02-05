# Data loader for IIIT5K dataset
# Handles loading images and labels for model training

import os
import numpy as np
import scipy.io as sio
from PIL import Image
import cv2
import config


def load_mat_data(mat_file):
    """
    Load data from MATLAB .mat file
    
    Args:
        mat_file: path to .mat file
        
    Returns:
        list of tuples (image_path, ground_truth)
    """
    data = sio.loadmat(mat_file)
    
    dataset = []
    
    # check which key exists (traindata or testdata)
    key = 'traindata' if 'traindata' in data else 'testdata'
    mat_data = data[key][0]
    
    for item in mat_data:
        # IIIT5K format uses named fields
        img_name = item['ImgName'][0]
        ground_truth = item['GroundTruth'][0]
        
        dataset.append((img_name, ground_truth))
    
    return dataset


def load_training_data():
    """Load training images and labels from IIIT5K dataset"""
    print("Loading training data...")
    
    train_data = load_mat_data(config.TRAIN_MAT)
    
    images = []
    labels = []
    
    for img_name, label in train_data:
        # image path is relative to dataset folder
        img_path = os.path.join(config.DATASET_DIR, img_name)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = preprocess_image(img)
                images.append(img)
                labels.append(label.lower())
    
    print(f"Loaded {len(images)} training images")
    return np.array(images), labels


def load_test_data():
    """Load test images and labels from IIIT5K dataset"""
    print("Loading test data...")
    
    test_data = load_mat_data(config.TEST_MAT)
    
    images = []
    labels = []
    
    for img_name, label in test_data:
        img_path = os.path.join(config.DATASET_DIR, img_name)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = preprocess_image(img)
                images.append(img)
                labels.append(label.lower())
    
    print(f"Loaded {len(images)} test images")
    return np.array(images), labels


def preprocess_image(img):
    """Preprocess image for model input"""
    # resize to fixed dimensions
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
    
    # normalize pixel values to 0-1
    img = img.astype(np.float32) / 255.0
    
    # add channel dimension
    img = np.expand_dims(img, axis=-1)
    
    return img


def encode_labels(labels):
    """
    Convert text labels to numeric sequences
    
    Args:
        labels: list of text strings
        
    Returns:
        numpy array of encoded labels
    """
    char_to_num = {char: idx for idx, char in enumerate(config.CHARACTERS)}
    
    encoded = []
    max_len = config.MAX_TEXT_LENGTH
    
    for label in labels:
        # convert characters to indices
        encoded_label = [char_to_num.get(char, 0) for char in label if char in char_to_num]
        
        # pad or truncate to max length
        if len(encoded_label) < max_len:
            encoded_label += [0] * (max_len - len(encoded_label))
        else:
            encoded_label = encoded_label[:max_len]
        
        encoded.append(encoded_label)
    
    return np.array(encoded)


def decode_predictions(predictions):
    """
    Convert numeric predictions back to text
    
    Args:
        predictions: numpy array of predicted indices
        
    Returns:
        list of decoded text strings
    """
    num_to_char = {idx: char for idx, char in enumerate(config.CHARACTERS)}
    
    decoded = []
    for pred in predictions:
        text = ''.join([num_to_char.get(int(idx), '') for idx in pred if idx > 0])
        decoded.append(text)
    
    return decoded


if __name__ == "__main__":
    # test data loading
    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Sample labels: {y_train[:5]}")
