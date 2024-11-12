import cv2
import numpy as np
import requests
import time
import base64
from threading import Thread
from queue import Queue

# Server address (change this to your actual server IP)
SERVER_URL = "http://192.168.151.120:8000/dehaze"

# Initialize camera
camera = cv2.VideoCapture(0)  # Index 0 is usually the first connected USB camera
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution (Optional: set to higher resolution)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Queue to store the latest dehazed image
dehazed_queue = Queue()

def send_image_to_server(img_encoded):
    """
    Sends the encoded image to the server for dehazing.
    
    Args:
    - img_encoded (np.ndarray): The encoded image.
    
    Returns:
    - response (requests.models.Response): The server response.
    """
    headers = {'Content-Type': 'multipart/form-data'}
    try:
        # Convert the image to a byte array and send it as a file in the POST request
        files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        response = requests.post(SERVER_URL, files=files)
        return response
    except Exception as e:
        print(f"Error sending image to server: {e}")
        return None

def capture_and_send_image():
    """
    Captures an image from the USB camera, sends it to the server for dehazing,
    and stores the dehazed image in a queue.
    """
    # Capture image from camera
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        return None

    # Resize the frame to the size expected by the model (224x224)
    haze_layer = np.ones_like(frame, dtype=np.float32) * 255
    hazy_image = cv2.addWeighted(frame, 1 - 0.3, haze_layer.astype(np.uint8), 0.3, 0)
    
    resized_frame = cv2.resize(hazy_image, (224, 224))

    # Encode the frame to send as a JPEG image
    _, img_encoded = cv2.imencode('.jpg', resized_frame)

    # Send the image to the server for dehazing
    response = send_image_to_server(img_encoded)
    dehazed_img = None

    if response and response.status_code == 200:
        response_data = response.json()
        dehazed_image_base64 = response_data.get('dehazed_image')

        # Decode the base64 image back to bytes
        img_data = base64.b64decode(dehazed_image_base64)
        # Convert bytes to numpy array
        np_arr = np.frombuffer(img_data, np.uint8)
        # Decode image from array
        dehazed_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Store the dehazed image in the queue
    dehazed_queue.put(dehazed_img)

    return hazy_image

def show_images():
    """
    Continuously captures frames, sends them for dehazing, and displays both the original and dehazed images.
    """
    while True:
        # Capture and process the image every 0.5 seconds
        resized_frame = capture_and_send_image()

        if resized_frame is not None:
            # Resize the original frame for display (if necessary)
            original_display = cv2.resize(resized_frame, (640, 360))  # Adjust this resolution as needed

            # Check if a dehazed image is available in the queue
            if not dehazed_queue.empty():
                dehazed_img = dehazed_queue.get()

                if dehazed_img is not None:
                    # Resize the dehazed image for display
                    dehazed_display = cv2.resize(dehazed_img, (640, 360))  # Adjust resolution as needed
                    
                    # Display the original and dehazed image side by side
                    combined_display = np.concatenate((original_display, dehazed_display), axis=1)
                    cv2.imshow('Original and Dehazed', combined_display)
                else:
                    # If dehazed image is not ready, just display the original frame
                    cv2.imshow('Original and Dehazed', original_display)
            else:
                # If no dehazed image, just show the original frame
                cv2.imshow('Original and Dehazed', original_display)

        # Wait for the user to press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the image display and processing in a separate thread to maintain real-time capture
    thread = Thread(target=show_images)
    thread.start()
