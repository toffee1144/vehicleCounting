from flask import Flask, Response
import cv2
from detection import frame_lock, current_frame
import mysql.connector

app = Flask(__name__)
appData = Flask(__name__)

def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if not ret:
                continue
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@appData.route('/data', methods=['GET'])
def get_vehicles():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="telkomiot123",
            database="AI_Vehicle"
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM counting")  # Adjust the table name if needed
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print("Error retrieving vehicle data from database:", e)
        return []
