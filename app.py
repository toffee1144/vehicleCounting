from flask import Flask, Response
import cv2
from detection import generate_frames
import mysql.connector

app = Flask(__name__)
appData = Flask(__name__)

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
