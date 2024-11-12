from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import base64
from app.DehazeModel import DehazeModel
from ultra_logger import Logger
import time


class DehazeServer:
    def __init__(self, dehaze_model: DehazeModel, logging=True, log_file=None):
        """
        Initializes the FastAPI server for image dehazing requests.
        
        Args:
        - dehaze_model (DehazeModel): An instance of DehazeModel to process requests.
        """
        self.dehaze_model = dehaze_model
        self.logging = logging
        self.app = FastAPI()
        self.setup_routes()
        
        if logging:
            log_file = './logs/DehazeServer.log' if not log_file else log_file
            self.logger = Logger('DehazeServer', log_file, clear_previous=True, log_to_console=False)
            
            # specify the time and the ip of the server
            self.logger.info('Initialized')        
    

    def setup_routes(self):
        """
        Sets up the server routes for dehazing.
        """
        @self.app.post("/dehaze")
        async def dehaze_image(file: UploadFile = File(...)):
            try:
                # start the timer
                start = time.time()
                
                img_array = np.frombuffer(await file.read(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if self.logging:
                    self.logger.info(f'Recieved image from client.')
                
                # Preprocess and predict
                img_tensor = self.dehaze_model.preprocess_image(img)
                dehazed_img_tensor = self.dehaze_model.predict(img_tensor)[0].numpy()
                dehazed_img = (dehazed_img_tensor * 255).astype(np.uint8)

                # Encode dehazed image to send as response
                _, buffer = cv2.imencode('.jpg', dehazed_img)

                # Convert the image to a base64 string
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                # show the img before sending
                # cv2.imshow('Dehazed Image', dehazed_img)
                # cv2.waitKey(1)
                # Return the base64-encoded image in the response

                resp = JSONResponse(content={"dehazed_image": img_base64})

                if self.logging:
                    self.logger.info(f'Image Dehazed in {time.time()-start} seconds. Sending now')
                
                return resp

            except Exception as e:
                self.logger.error(f'Error encountered while processing the image from client: {e}')
                return JSONResponse(status_code=500)
                
            finally:
                pass                
                
    def run(self, host='0.0.0.0', port=8000):
        """
        Runs the server on the specified host and port.
        
        Args:
        - host (str): Host IP address.
        - port (int): Port number.
        """
        
        if self.logging:
            self.logger.info('Starting server')

        uvicorn.run(self.app, host=host, port=port)
