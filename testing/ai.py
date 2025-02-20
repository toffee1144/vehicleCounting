import cv2
import argparse
import numpy as np
import gi
import sys
import os
import time
import threading
from flask import Flask, Response
import hailo
import mysql.connector  # pip install mysql-connector-python
import datetime
import pytz  # install via pip if needed: pip install pytz

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject

app = Flask(__name__)
frame_lock = threading.Lock()
current_frame = None
jakarta_tz = pytz.timezone("Asia/Jakarta")
POLYGON_1 = np.array([(61, 438), (241, 397), (373, 616), (145, 620)], np.int32)
previous_centers = {}
counted_ids = {}
rtsp_link = None

# Global buffer to store data if database connection fails.
offline_data = []

def flush_offline_data(cursor, conn):
    """Attempt to flush any data stored offline to the database."""
    global offline_data
    for data in offline_data:
        sql = """
            INSERT INTO counting (car, motorcycle, bus, truck, time)
            VALUES (%s, %s, %s, %s, %s)
        """
        values = (data['car'], data['motorcycle'], data['bus'], data['truck'], data['time'])
        cursor.execute(sql, values)
        conn.commit()
    if offline_data:
        print("Flushed offline data:", offline_data)
    offline_data = []

def handle_database_error(car, motorcycle, bus, truck, current_timestamp, ):
    """Store the current counts in an offline buffer for later insertion."""
    global offline_data
    data = {
        'car': car,
        'motorcycle': motorcycle,
        'bus': bus,
        'truck': truck,
        'time': current_timestamp
    }
    offline_data.append(data)
    print("Stored data offline due to database error:", data)

class HailoDetectionApp:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.detections = []
        self.running = True
        self.hef_path = "/home/pi/Documents/hailo-rpi5-examples/basic_pipelines/yolov11s.hef"
        self.post_process_so = "/home/pi/Videos/OpenCV/libyolo_hailortpp_postprocess.so"
        self.nms_score_threshold = 0.4  # Adjust threshold as needed
        self.nms_iou_threshold = 0.5    # Adjust threshold as needed
        self.tracked_classes = {4: "motorcycle", 8: "truck", 6: "bus", 3: "car"}
        self.last_time = time.time()
        self.fps = 0
        self.car_crossed_count = 0  
        self.truck_crossed_count = 0  
        self.bus_crossed_count = 0  
        self.motorcycle_crossed_count = 0
        self.frame_count = 0  
        self.blink_status = False  
        self.last_detection_time = 0
        self.blink_duration = 0.15  
        self.blink_interval = 0.15  

        Gst.init(None)
        self.create_pipeline()

    def create_pipeline(self):
        pipeline_str = (
            f'rtspsrc location={self.rtsp_url} latency=0 ! '
            'queue name=source_queue_decode leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'rtph264depay ! h264parse ! avdec_h264 ! '
            'queue name=source_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'videoscale name=source_videoscale n-threads=2 ! '
            'queue name=source_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'videoconvert n-threads=3 name=source_convert qos=false ! '
            'video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=1280, height=720 ! '
            'queue name=inference_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so '
            'function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true '
            'hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! '
            'queue name=inference_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! '
            'inference_wrapper_agg.sink_0 inference_wrapper_crop. ! '
            'queue name=inference_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'videoscale name=inference_videoscale n-threads=2 qos=false ! '
            'queue name=inference_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'video/x-raw, pixel-aspect-ratio=1/1 ! '
            'videoconvert name=inference_videoconvert n-threads=2 ! '
            'queue name=inference_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet name=inference_hailonet hef-path={self.hef_path} batch-size=2 vdevice-group-id=1 '
            f'nms-score-threshold={self.nms_score_threshold} nms-iou-threshold={self.nms_iou_threshold} '
            'output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true ! '
            'queue name=inference_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter name=inference_hailofilter so-path={self.post_process_so} function-name=filter_letterbox qos=false ! '
            'queue name=inference_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'inference_wrapper_agg.sink_1 inference_wrapper_agg. ! '
            'queue name=inference_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'hailotracker name=hailo_tracker class-id=-1 kalman-dist-thr=0.6 iou-thr=0.95 init-iou-thr=0.8 '
            'keep-new-frames=2 keep-tracked-frames=5 keep-lost-frames=3 keep-past-metadata=False qos=False ! '
            'queue name=hailo_tracker_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'queue name=identity_callback_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'identity name=identity_callback ! '
            'queue name=hailo_display_videoconvert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! '
            'queue name=hailo_display_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            'appsink name=appsink emit-signals=True sync=False qos=False max-buffers=1 drop=True'
        )

        self.pipeline = Gst.parse_launch(pipeline_str)
        appsink = self.pipeline.get_by_name('appsink')
        appsink.connect('new-sample', self.on_new_sample)

        # Set up bus message handling to catch RTSP errors.
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_bus_message)
        self.loop = GLib.MainLoop()

    def on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("RTSP Error:", err, debug)
            self.handle_rtsp_error(err)

    def handle_rtsp_error(self, error):
        """Handle errors from the RTSP source by attempting to restart the pipeline."""
        print("Handling RTSP error, attempting to restart pipeline in 5 seconds...")
        time.sleep(5)
        self.pipeline.set_state(Gst.State.NULL)
        self.pipeline.set_state(Gst.State.PLAYING)

    def is_inside_polygon(self, bbox, polygon):
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        points = [
            (x_min, y_min),
            (x_max, y_min),
            (x_min, y_max),
            (x_max, y_max),
            (center_x, center_y)
        ]

        for point in points:
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                return True
        return False

    def on_new_sample(self, sink):
        global current_frame
        sample = sink.emit('pull-sample')
        if not sample:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        if not buffer:
            return Gst.FlowReturn.ERROR

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            frame = np.ndarray(
                shape=(720, 1280, 3),
                dtype=np.uint8,
                buffer=map_info.data
            ).copy()

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            current_time = time.time()
            self.fps = 1 / (current_time - self.last_time)
            self.last_time = current_time

            # Get detections via Hailo SDK.
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            self.detections.clear()

            for detection in detections:
                label_id = detection.get_class_id()
                if label_id not in self.tracked_classes:
                    continue
                bbox = detection.get_bbox()
                confidence = detection.get_confidence()
                label = self.tracked_classes[label_id]
                track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                object_id = track[0] if track else None
                x1, y1, x2, y2 = map(int, [bbox.xmin() * 1280, bbox.ymin() * 720, bbox.xmax() * 1280, bbox.ymax() * 720])
                
                if object_id is not None:
                    self.detections.append((x1, y1, x2, y2, label, confidence, object_id))

                if self.is_inside_polygon((x1, y1, x2, y2), POLYGON_1):
                    if object_id not in counted_ids:
                        counted_ids[object_id] = True
                        self.blink_status = True
                        self.last_detection_time = time.time()
                        if "car" in label.lower():
                            self.car_crossed_count += 1
                        elif "truck" in label.lower():
                            self.truck_crossed_count += 1
                        elif "bus" in label.lower():
                            self.bus_crossed_count += 1
                        elif "motorcycle" in label.lower():
                            self.motorcycle_crossed_count += 1

            if self.frame_count % 100 == 0:
                expired_ids = [obj_id for obj_id, ts in previous_centers.items() if current_time - ts > 5]
                for obj_id in expired_ids:
                    previous_centers.pop(obj_id, None)
                    counted_ids.pop(obj_id, None)
            self.frame_count += 1
            self.draw_detections(frame)

            # Display counts on the frame.
            cv2.putText(frame, f"Cars: {self.car_crossed_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Trucks: {self.truck_crossed_count}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Buses: {self.bus_crossed_count}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Motorcycles: {self.motorcycle_crossed_count}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Dynamic polygon coloring with blinking.
            elapsed = time.time() - self.last_detection_time
            if elapsed < self.blink_duration:
                num_intervals = int(elapsed / self.blink_interval)
                blink_state = num_intervals % 2
                polygon_color = (0, 0, 255) if blink_state == 0 else (0, 255, 0)
            else:
                polygon_color = (0, 255, 0)
            cv2.polylines(frame, [POLYGON_1], isClosed=True, color=polygon_color, thickness=2)

            with frame_lock:
                current_frame = frame
        finally:
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    def draw_detections(self, frame):
        for x1, y1, x2, y2, label, score, object_id in self.detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        fps_text = f"FPS: {self.fps:.2f}"
        text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = frame.shape[0] - 10
        cv2.putText(frame, fps_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()

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

def mysql_insertion_loop(app_instance):
    while True:
        time.sleep(5)
        car = app_instance.car_crossed_count
        motorcycle = app_instance.motorcycle_crossed_count
        bus = app_instance.bus_crossed_count
        truck = app_instance.truck_crossed_count
        current_timestamp = datetime.datetime.now(jakarta_tz)
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="telkomiot123",
                database="AI_Vehicle"
            )
            cursor = conn.cursor()
            # Flush any stored offline data.
            if offline_data:
                flush_offline_data(cursor, conn)
            cursor.execute("SELECT id, DATE(time) FROM counting ORDER BY time DESC LIMIT 1")
            result = cursor.fetchone()
            if result:
                last_date = result
                if last_date == current_timestamp.date():
                    sql = """
                        INSERT counting
                        SET car = car + %s,
                            motorcycle = motorcycle + %s,
                            bus = bus + %s,
                            truck = truck + %s,
                            time = %s
                    """
                    values = (car, motorcycle, bus, truck, current_timestamp)
                    cursor.execute(sql, values)
                else:
                    sql = """
                        INSERT INTO counting ( car, motorcycle, bus, truck, time)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    values = (car, motorcycle, bus, truck, current_timestamp)
                    cursor.execute(sql, values)
            else:
                sql = """
                    INSERT INTO counting (car, motorcycle, bus, truck, time)
                    VALUES (%s, %s, %s, %s, %s)
                """
                values = (car, motorcycle, bus, truck, current_timestamp)
                cursor.execute(sql, values)
            conn.commit()
            print("Updated data in MySQL:", car, motorcycle, bus, truck, "at", current_timestamp)
            cursor.close()
            conn.close()
            # Reset counts after a successful update.
            app_instance.car_crossed_count = 0
            app_instance.motorcycle_crossed_count = 0
            app_instance.bus_crossed_count = 0
            app_instance.truck_crossed_count = 0
        except Exception as e:
            handle_database_error(car, motorcycle, bus, truck, current_timestamp)

if __name__ == '__main__':
    def get_rtsp_link_from_db():
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="telkomiot123",
                database="AI_Vehicle"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT link FROM rtspLink LIMIT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            if result:
                return result[0]
            else:
                print("No RTSP link found in the database.")
                return None
        except Exception as e:
            print("Error retrieving RTSP link from database:", e)
            return None

    while rtsp_link is None:
        rtsp_link = get_rtsp_link_from_db()
        if rtsp_link is None:
            print("No RTSP link found. Waiting for the data...")
            time.sleep(5)  # Wait 5 seconds before trying again
            
    hailo_app = HailoDetectionApp(rtsp_link)
    hailo_thread = threading.Thread(target=hailo_app.run, daemon=True)
    hailo_thread.start()
    mysql_thread = threading.Thread(target=mysql_insertion_loop, args=(hailo_app,), daemon=True)
    mysql_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
