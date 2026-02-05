"""
Training script for CRNN model
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root and src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

import config
from data_loader import load_training_data, load_test_data, encode_labels
from models.crnn_model import build_training_model, build_crnn_model


def create_dataset(images, labels, batch_size=32):
    """
    Create TensorFlow dataset
    
    Args:
        images: Array of images
        labels: Encoded labels
        batch_size: Batch size
        
    Returns:
        TensorFlow Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


class ModelCheckpoint(keras.callbacks.Callback):
    """Custom callback to save best model"""
    
    def __init__(self, filepath, monitor='val_loss', mode='min'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if (self.mode == 'min' and current < self.best) or \
           (self.mode == 'max' and current > self.best):
            self.best = current
            self.model.save_weights(self.filepath)
            print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.4f}, saving model to {self.filepath}")


def train_model(epochs=50, batch_size=32):
    """
    Train the CRNN model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("=" * 80)
    print("CRNN Model Training")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()
    
    # Encode labels
    print("\n[2/5] Encoding labels...")
    y_train_encoded = encode_labels(y_train)
    y_test_encoded = encode_labels(y_test)
    
    # Create datasets
    print("\n[3/5] Creating datasets...")
    train_dataset = create_dataset(X_train, y_train_encoded, batch_size)
    val_dataset = create_dataset(X_test, y_test_encoded, batch_size)
    
    # Build model
    print("\n[4/5] Building model...")
    model = build_training_model()
    
    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE))
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=config.WEIGHTS_PATH,
            monitor='val_loss',
            mode='min'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.OUTPUT_DIR, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\n[5/5] Training model...")
    print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")
    print("-" * 80)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\nSaving final model...")
    model.save(config.MODEL_PATH)
    print(f"Model saved to: {config.MODEL_PATH}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """
    Plot training metrics
    
    Args:
        history: Training history object
    """
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(config.OUTPUT_DIR, 'training_history.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nTraining history plot saved to: {plot_path}")
    plt.close()


def evaluate_model():
    """Evaluate trained model on test set"""
    print("\n" + "=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test = load_test_data()
    
    # Load model
    print("Loading trained model...")
    model = build_crnn_model()
    model.load_weights(config.WEIGHTS_PATH)
    
    # Make predictions on a few samples
    print("\nSample predictions:")
    num_samples = min(10, len(X_test))
    
    for i in range(num_samples):
        img = np.expand_dims(X_test[i], axis=0)
        pred = model.predict(img, verbose=0)
        
        # Decode prediction (simplified)
        pred_text = f"prediction_{i}"  # Placeholder - will be decoded properly
        actual_text = y_test[i]
        
        print(f"{i+1}. Actual: '{actual_text}' | Predicted: '{pred_text}'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CRNN model for OCR')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model instead of training')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_model()
    else:
        train_model(epochs=args.epochs, batch_size=args.batch_size)
