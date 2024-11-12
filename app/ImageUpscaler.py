# import cv2
import cv2; print(cv2.__version__)

from cv2 import dnn_superres
import numpy as np
import requests
from PIL import Image
import base64
import torch
# from realesrgan import RealESRGANer

class ImageUpscaler:
    def __init__(self, model_type="fsrcnn", model_path=None):
        """
        Initializes the ImageUpscaler class with the specified model.

        Args:
        - model_type (str): The type of model to use ("fsrcnn", "srcnn", "lapsrn", or "esrgan").
        - model_path (str): Path to the model file, if needed.
        """
        self.model_type = model_type.lower()
        self.model = None

        if self.model_type in ["fsrcnn", "srcnn", "lapsrn"]:
            if model_path:
                self.load_cv2_model(model_path)
            else:
                raise ValueError("For FSRCNN, SRCNN, and LapSRN, please provide a model path.")
        # elif self.model_type == "esrgan":
            # self.model = RealESRGANer("RealESRGAN_x4.pth") if model_path is None else RealESRGANer(model_path)
        else:
            raise ValueError("Unsupported model type. Choose from 'fsrcnn', 'srcnn', 'lapsrn', or 'esrgan'.")

    def load_cv2_model(self, model_path):
        """
        Loads an OpenCV DNN Super-Resolution model (FSRCNN, SRCNN, or LapSRN).

        Args:
        - model_path (str): Path to the .pb file for FSRCNN, SRCNN, or LapSRN.
        """
        self.model = dnn_superres.DnnSuperResImpl()
        scale_factor = 4  # Adjust based on model
        self.model.readModel(model_path)
        self.model.setModel(self.model_type, scale_factor)

    def preprocess_image(self, img_path):
        """
        Loads and preprocesses an image from a file path.

        Args:
        - img_path (str): Path to the input image.

        Returns:
        - np.array: Preprocessed image array.
        """
        img = cv2.imread(img_path)
        return img

    def upscale_image(self, img_array):
        """
        Upscales the input image based on the loaded model.

        Args:
        - img_array (np.array): Input image array to be upscaled.

        Returns:
        - np.array: Upscaled image.
        """
        if self.model_type in ["fsrcnn", "srcnn", "lapsrn"]:
            return self.model.upsample(img_array)
        elif self.model_type == "esrgan":
            pil_image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            upscaled_pil_image = self.model.predict(pil_image)
            return cv2.cvtColor(np.array(upscaled_pil_image), cv2.COLOR_RGB2BGR)

    def save_image(self, img_array, save_path):
        """
        Saves an image array to the specified path.

        Args:
        - img_array (np.array): Image array to save.
        - save_path (str): Path where the image will be saved.
        """
        cv2.imwrite(save_path, img_array)


# Example usage:
if __name__ == "__main__":
    # FSRCNN example
    # fsrcnn_path = "../model/FSRCNN/FSRCNN_x2.pb"  # Example path
    # upscaler_fsrcnn = ImageUpscaler(model_type="fsrcnn", model_path=fsrcnn_path)
    # input_image = upscaler_fsrcnn.preprocess_image("low_res_image.jpg")
    # upscaled_image_fsrcnn = upscaler_fsrcnn.upscale_image(input_image)
    # upscaler_fsrcnn.save_image(upscaled_image_fsrcnn, "upscaled_fsrcnn.jpg")

    # SRCNN example
    # srcnn_path = "../model/SRCNN/SRCNN.pb"  # Example path
    # upscaler_srcnn = ImageUpscaler(model_type="srcnn", model_path=srcnn_path)
    # input_image = upscaler_srcnn.preprocess_image("low_res_image.jpg")
    # upscaled_image_srcnn = upscaler_srcnn.upscale_image(input_image)
    # upscaler_srcnn.save_image(upscaled_image_srcnn, "upscaled_srcnn.jpg")

    # LapSRN example
    lapsrn_path = "LapSRN_x8.pb"  # Example path
    upscaler_lapsrn = ImageUpscaler(model_type="lapsrn", model_path=lapsrn_path)
    input_image = upscaler_lapsrn.preprocess_image("low_res_image.jpg")
    upscaled_image_lapsrn = upscaler_lapsrn.upscale_image(input_image)
    upscaler_lapsrn.save_image(upscaled_image_lapsrn, "upscaled_lapsrn.jpg")

    # ESRGAN example
    # esrgan_path = "../model/ESRGAN/RealESRGAN_x4plus_anime_6B.pth"  # Example path
    # upscaler_esrgan = ImageUpscaler(model_type="esrgan", model_path=esrgan_path)
    # input_image = upscaler_esrgan.preprocess_image("low_res_image.jpg")
    # upscaled_image_esrgan = upscaler_esrgan.upscale_image(input_image)
    # upscaler_esrgan.save_image(upscaled_image_esrgan, "upscaled_esrgan.jpg")
