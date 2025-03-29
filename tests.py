import tensorflow as tf
import numpy as np
import cv2

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

    def apply_clahe(self, img_array):
            """
            Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image to enhance local contrast and
            reduce over-exposure or under-exposure in the image, especially useful for dehazing.

            Args:
            - img_array (np.array): Image to apply CLAHE to.

            Returns:
            - np.array: Image with CLAHE applied.
            """
            # Convert image from RGB to LAB for CLAHE application (CLAHE works on L channel)
            lab_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab_img)

            # Apply CLAHE to the L-channel (Lightness)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_planes[0] = clahe.apply(lab_planes[0])  # CLAHE on the L-channel (Lightness)
            lab_img = cv2.merge(lab_planes)

            # Convert back to BGR color space
            enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
            return enhanced_img

    def preprocess_image(self, img):
        """
        Preprocesses an image (from a path or array) for model input using CLAHE to enhance contrast.
        
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

        # Apply CLAHE as part of preprocessing
        img = self.apply_clahe(img.numpy())  # Convert Tensor to numpy array for CLAHE

        # Resize and normalize the image for model input
        img = tf.convert_to_tensor(img, dtype=tf.float32)  # Convert back to tensor after CLAHE
        img = tf.image.resize(img, (224, 224)) / 255.0

        return tf.expand_dims(img, axis=0)  # Add batch dimension

    def preprocess_image_from_array(self, img_array):
        """
        Preprocesses an image directly from a NumPy array for prediction, with CLAHE applied.

        Args:
        - img_array (np.array): Image array captured from camera.

        Returns:
        - tf.Tensor: Preprocessed image tensor.
        """
        # Apply CLAHE as part of preprocessing
        img_array = self.apply_clahe(img_array)

        # Convert the processed image to a tensor and resize
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32) / 255.0
        img_tensor = tf.image.resize(img_tensor, (224, 224))
        return tf.expand_dims(img_tensor, axis=0)  # Add batch dimension

    def postprocess_image(self, img_array):
        """
        Post-processes the image to enhance contrast and brightness, and reduce errors like mistakenly darkened areas.

        Args:
        - img_array (np.array): Image array to post-process.

        Returns:
        - np.array: Post-processed image array.
        """
        # Convert image from RGB to LAB for CLAHE application
        lab_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab_img)

        # Apply CLAHE to L channel to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])  # Apply CLAHE on the L-channel (Lightness)
        lab_img = cv2.merge(lab_planes)

        # Convert back to BGR color space
        enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

        # Apply additional contrast and brightness correction
        contrast_factor = 1.5
        brightness_factor = 30
        adjusted_img = cv2.convertScaleAbs(enhanced_img, alpha=contrast_factor, beta=brightness_factor)

        return adjusted_img


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
