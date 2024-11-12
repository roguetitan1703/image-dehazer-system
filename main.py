import sys
from app.DehazeModel import DehazeModel
from app.DehazeServer import DehazeServer
from app.CameraDehazer import CameraDehazer

def run_server():
    model_path = "model/gman_net_model.h5"
    dehaze_model = DehazeModel(model_path=model_path)
    server = DehazeServer(dehaze_model=dehaze_model)
    server.run(host='0.0.0.0', port=8000)

def run_camera_dehazer():
    model_path = "model/gman_net_model.h5"
    dehaze_model = DehazeModel(model_path=model_path)
    camera_dehazer = CameraDehazer(dehaze_model=dehaze_model, capture_interval=0, haze_intensity=0.2)
    camera_dehazer.start_realtime_dehazing()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [server|camera]")
    elif sys.argv[1] == "server":
        run_server()
    elif sys.argv[1] == "camera":
        run_camera_dehazer()
    else:
        print("Invalid option. Use 'server' to run the server or 'camera' for real-time dehazing.")
