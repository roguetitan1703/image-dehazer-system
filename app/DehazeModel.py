import tensorflow as tf
import numpy as np

class DehazeModel:
    def __init__(self, model_path=None):
        """
        Initializes the DehazeModel class, loading the model if a model path is provided.
        
        Args:
        - model_path (str): Path to the saved model.
        """
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Loads the model architecture and weights.
        
        Args:
        - model_path (str): Path to the saved model file.
        """
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, img):
        """
        Preprocesses an image (from a path or array) for model input.
        
        Args:
        - img (str or np.array): Path to the image or image array.
        
        Returns:
        - tf.Tensor: Preprocessed image tensor ready for prediction.
        """
        if isinstance(img, str):  # If img is a file path
            img = tf.io.read_file(img)
            img = tf.io.decode_jpeg(img, channels=3)
        elif isinstance(img, np.ndarray):  # If img is an array
            img = tf.convert_to_tensor(img, dtype=tf.float32)

        img = tf.image.resize(img, (224, 224)) / 255.0
        return tf.expand_dims(img, axis=0)

    def preprocess_image_from_array(self, img_array):
        """
        Preprocesses an image directly from a NumPy array for prediction.

        Args:
        - img_array (np.array): Image array captured from camera.

        Returns:
        - tf.Tensor: Preprocessed image tensor.
        """
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32) / 255.0
        img_tensor = tf.image.resize(img_tensor, (224, 224))
        return tf.expand_dims(img_tensor, axis=0)  # Add batch dimension

    def predict(self, img_tensor):
        """
        Predicts the dehazed image for the given preprocessed image tensor.
        
        Args:
        - img_tensor (tf.Tensor): Preprocessed image tensor.
        
        Returns:
        - tf.Tensor: Predicted dehazed image tensor.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        return self.model(img_tensor, training=False)
