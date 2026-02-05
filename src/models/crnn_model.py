"""
CRNN (Convolutional Recurrent Neural Network) Model for OCR
Architecture: CNN + RNN + CTC Loss
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


def build_crnn_model():
    """
    Build CRNN model architecture
    
    Returns:
        Keras model
    """
    # Input layer
    input_img = layers.Input(
        shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1),
        name='image_input'
    )
    
    # Convolutional layers for feature extraction
    x = input_img
    
    # Conv Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # Conv Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # Conv Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = layers.MaxPooling2D((2, 1), name='pool3')(x)  # Pool only height
    
    # Conv Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4_1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = layers.BatchNormalization(name='bn4_2')(x)
    x = layers.MaxPooling2D((2, 1), name='pool4')(x)
    
    # Conv Block 5
    x = layers.Conv2D(512, (2, 2), activation='relu', padding='same', name='conv5')(x)
    
    # Reshape for RNN
    # After conv layers: height reduces from 32 to 2 (32/16), width from 128 to 32 (128/4)
    # Pooling: pool1(2,2) pool2(2,2) pool3(2,1) pool4(2,1) = height/16, width/4
    # Shape: (batch, 2, 32, 512) -> (batch, 32, 2*512) = (batch, 32, 1024)
    x = layers.Reshape(target_shape=(config.IMG_WIDTH // 4, -1), name='reshape')(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    
    # Recurrent layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25), name='birnn1')(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25), name='birnn2')(x)
    
    # Output layer
    x = layers.Dense(config.NUM_CLASSES, activation='softmax', name='dense2')(x)
    
    # Create model
    model = keras.Model(inputs=input_img, outputs=x, name='CRNN')
    
    return model


def ctc_loss_function(y_true, y_pred):
    """
    CTC (Connectionist Temporal Classification) loss function
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        CTC loss
    """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


class CTCLayer(layers.Layer):
    """Custom CTC layer for training"""
    
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_loss_function

    def call(self, y_true, y_pred):
        # Compute the CTC loss
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len,), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len,), dtype="int64")

        loss = self.loss_fn(y_true, y_pred)
        self.add_loss(loss)

        return y_pred


def build_training_model():
    """
    Build model with CTC layer for training
    
    Returns:
        Training model
    """
    # Build base model
    model = build_crnn_model()
    
    # Add CTC layer
    labels = layers.Input(name="label", shape=(None,), dtype="float32")
    output = CTCLayer(name="ctc_loss")(labels, model.output)
    
    # Create training model
    training_model = keras.Model(
        inputs=[model.input, labels],
        outputs=output
    )
    
    return training_model


def decode_batch_predictions(pred):
    """
    Decode predictions using greedy search
    
    Args:
        pred: Model predictions
        
    Returns:
        Decoded text
    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    # Use greedy search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :config.MAX_TEXT_LENGTH]
    
    # Decode to text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    
    return output_text


if __name__ == "__main__":
    # Test model building
    import numpy as np
    
    print("Building CRNN model...")
    model = build_crnn_model()
    model.summary()
    
    print("\nBuilding training model...")
    training_model = build_training_model()
    training_model.summary()
    
    # Test with random input
    test_input = np.random.rand(1, config.IMG_HEIGHT, config.IMG_WIDTH, 1)
    test_label = np.random.randint(0, config.NUM_CLASSES, (1, config.MAX_TEXT_LENGTH))
    
    print("\nTesting model with random input...")
    output = model.predict(test_input)
    print(f"Output shape: {output.shape}")
