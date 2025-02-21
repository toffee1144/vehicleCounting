import threading
import time
import cv2
from detection import HailoDetectionApp
from database import get_rtsp_link_from_db, mysql_insertion_loop, send_data_via_mqtt
from app import app as video_app, appData as data_app

def run_video_app():
    # Run the video streaming Flask app on port 5000.
    video_app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

def run_data_app():
    # Run the data API Flask app on port 5001.
    data_app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)

def is_rtsp_link_accessible(rtsp_link):
    cap = cv2.VideoCapture(rtsp_link)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    cap.release()
    return ret

def main():
    # Get the RTSP link from the database.
    rtsp_link = None
    while rtsp_link is None:
        rtsp_link = get_rtsp_link_from_db()
        if rtsp_link is None:
            print("No RTSP link found. Waiting for the data...")
            time.sleep(5)

    while not is_rtsp_link_accessible(rtsp_link):
        print("RTSP link is not accessible. Waiting for the data...")
        time.sleep(5)

    # Start the Hailo detection process.
    hailo_app = HailoDetectionApp(rtsp_link)
    hailo_thread = threading.Thread(target=hailo_app.run, daemon=True)
    hailo_thread.start()

    # Start the video streaming Flask app on port 5000.
    video_thread = threading.Thread(target=run_video_app, daemon=True)
    video_thread.start()

    # Start the MySQL insertion thread.
    mysql_thread = threading.Thread(target=mysql_insertion_loop, args=(hailo_app,), daemon=True)
    mysql_thread.start()

    # Start the data API Flask app on port 5001.
    data_thread = threading.Thread(target=run_data_app, daemon=True)
    data_thread.start()

    # Start the MQTT data sending thread.
    mqtt_thread = threading.Thread(target=send_data_via_mqtt, daemon=True)
    mqtt_thread.start()

    # Keep the main thread alive.
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()