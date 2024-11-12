import cv2
import numpy as np
import time
from threading import Thread
from app.DehazeModel import DehazeModel

class CameraDehazer:
    def __init__(self, dehaze_model: DehazeModel, capture_interval=3, haze_intensity=0.5):
        """
        Initializes the CameraDehazer with a dehazing model and sets up the camera feed.

        Args:
        - dehaze_model (DehazeModel): An instance of DehazeModel for processing frames.
        - capture_interval (int): Time in seconds between each dehazed frame capture.
        - haze_intensity (float): Intensity of the haze effect (0.0 to 1.0).
        """
        self.dehaze_model = dehaze_model
        self.capture_interval = capture_interval
        self.haze_intensity = haze_intensity
        self.camera = cv2.VideoCapture(0)
        
        # Initialize threading attributes
        self.last_capture_time = time.time()
        self.running = True
        
        self.dehaze_thread = Thread(target=self._dehaze_loop)
        self.dehaze_thread.start()

    def _dehaze_loop(self):
        """
        Continuously captures and dehazes images at the specified interval.
        """
        while self.running:
            current_time = time.time()
            if current_time - self.last_capture_time >= self.capture_interval:
                ret, frame = self.camera.read()
                if ret:
                    hazed_frame = self.add_uneven_haze(frame, self.haze_intensity)
                    img_tensor = self.dehaze_model.preprocess_image_from_array(hazed_frame)
                    self.dehazed_image = self.dehaze_model.predict(img_tensor)[0].numpy()
                    self.last_capture_time = current_time
    
    def start_realtime_dehazing(self):
        """
        Starts real-time dehazing using the camera feed and OpenCV.
        """
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                break

            # Resize the frame to the same size as the model's expected input
            resized_frame = cv2.resize(frame, (224, 224))

            # Add haze to the resized frame
            hazy_frame = self.add_haze(resized_frame, self.haze_intensity)

            # Preprocess the hazy frame for prediction
            img_tensor = self.dehaze_model.preprocess_image_from_array(hazy_frame)
            dehazed_frame = self.dehaze_model.predict(img_tensor)[0].numpy()

            # Display dehazed frame (scaled back to original dimensions if needed)
            dehazed_display = (dehazed_frame * 255).astype(np.uint8)

            # Resize dehazed frame to match the original input size for concatenation
            dehazed_display_resized = cv2.resize(dehazed_display, (resized_frame.shape[1], resized_frame.shape[0]))

            # Display hazy and dehazed frames side by side
            combined_display = np.concatenate((hazy_frame, dehazed_display_resized), axis=1)
            cv2.imshow('Hazy Frame and Dehazed Frame', combined_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def add_haze(self, image, haze_intensity=0.5):
        """
        Adds a simple haze effect to an image.
        """
        haze_layer = np.ones_like(image, dtype=np.float32) * 255
        hazy_image = cv2.addWeighted(image, 1 - haze_intensity, haze_layer.astype(np.uint8), haze_intensity, 0)
        return hazy_image

    def add_uneven_haze(self, image, haze_intensity=0.8):
        """
        Adds uneven haze to an image based on a randomly generated mask.
        """
        haze_layer = np.random.randint(200, 255, size=image.shape, dtype=np.uint8)
        noise_mask = np.random.rand(*image.shape[:2]) * 255
        noise_mask = cv2.GaussianBlur(noise_mask.astype(np.uint8), (21, 21), 0)
        normalized_mask = noise_mask.astype(np.float32) / 255.0
        beta = haze_intensity * np.mean(normalized_mask)
        hazy_image = cv2.addWeighted(image, 1 - haze_intensity, haze_layer, beta, 0)
        return hazy_image

    def stop(self):
        """
        Stops the real-time dehazing and releases the camera.
        """
        self.running = False
        self.dehaze_thread.join()
        self.camera.release()
        cv2.destroyAllWindows()
