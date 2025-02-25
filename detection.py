import cv2
import time
import numpy as np
import threading
import gi
import hailo  # Make sure the Hailo SDK is installed and accessible
from devInfo import delete_hls_file
import logging

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject

# Global variables used for detection.
POLYGON_1 = np.array([(234, 362), (396, 341), (436, 521), (222, 509)], np.int32)
counted_ids = {}
previous_centers = {}

# These globals will be used by both detection and the Flask app.
frame_lock = threading.Lock()
current_frame = None

class HailoDetectionApp:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.detections = []
        self.running = True
        self.hef_path = "/home/pi/Documents/hailo-rpi5-examples/basic_pipelines/yolov11s.hef"
        self.post_process_so = "/home/pi/Videos/OpenCV/libyolo_hailortpp_postprocess.so"
        self.nms_score_threshold = 0.4
        self.nms_iou_threshold = 0.5
        self.tracked_classes = {4: "motorcycle", 8: "truck", 6: "bus", 3: "car"}
        self.last_time = time.time()
        self.fps = 0
        self.car_crossed_count = 0  
        self.truck_crossed_count = 0  
        self.bus_crossed_count = 0  
        self.motorcycle_crossed_count = 0
        self.frame_count = 0  
        self.reconnecting = False  # Flag to track reconnection status

        # For blinking on a new count (blink only one frame)
        self.blink_once = False

        Gst.init(None)
        self.create_pipeline()
            
    def get_vehicle_data(self):
        """Return the current vehicle counts."""
        return {
            "car": self.car_crossed_count,
            "truck": self.truck_crossed_count,
            "bus": self.bus_crossed_count,
            "motorcycle": self.motorcycle_crossed_count
        }

    def create_pipeline(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.set_state(Gst.State.NULL)
            del self.pipeline

        global counted_ids, previous_centers
        counted_ids.clear()
        previous_centers.clear()

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
            print(f"RTSP Error: {err} - {debug}")
            if not self.reconnecting:
                GLib.idle_add(self.handle_error_and_reconnect)

    def handle_error_and_reconnect(self):
        print("Handling error and reconnecting...")
        self.reconnecting = True
        self.pipeline.set_state(Gst.State.NULL)
        self.schedule_reconnect()
        return False

    def schedule_reconnect(self):
        GLib.timeout_add(5000, self.try_reconnect)

    def try_reconnect(self):
        print("Attempting to reconnect...")
        try:
            self.create_pipeline()  # Recreate the pipeline
            self.pipeline.set_state(Gst.State.PLAYING)
            self.reconnecting = False
            return False
        except Exception as e:
            print(f"Reconnection failed: {str(e)}")
            return True  # Continue retrying

    def is_inside_polygon(self, bbox, polygon):
        """Check if any key point of a bounding box is inside the polygon."""
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

    def get_polygon_color(self):
        """
        Return a polygon color.
        If a new detection has been counted (blink_once is True),
        return the blink color and then reset the flag.
        Otherwise, return the normal color.
        """
        if self.blink_once:
            # Reset the flag so that the blink shows only one frame.
            self.blink_once = False
            return (0, 0, 255)  # Blink color (red)
        return (0, 255, 0)      # Default color (green)

    def on_new_sample(self, sink):
        if self.reconnecting or not self.pipeline:
            return Gst.FlowReturn.OK
        
        global current_frame, frame_lock, counted_ids, previous_centers
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
            current_state = self.pipeline.get_state(0).state
            if current_state != Gst.State.PLAYING:
                return Gst.FlowReturn.OK

            # Process frame only if pipeline is active.
            frame = np.ndarray(
                shape=(720, 1280, 3),
                dtype=np.uint8,
                buffer=map_info.data
            ).copy()

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            current_time = time.time()
            self.fps = 1 / (current_time - self.last_time)
            self.last_time = current_time

            # Get detections via the Hailo SDK.
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
                x1, y1, x2, y2 = map(int, [bbox.xmin() * 1280, bbox.ymin() * 720,
                                            bbox.xmax() * 1280, bbox.ymax() * 720])
                
                if object_id is not None:
                    self.detections.append((x1, y1, x2, y2, label, confidence, object_id))

                # When a new detection is inside the polygon, update counts and trigger a one-time blink.
                if self.is_inside_polygon((x1, y1, x2, y2), POLYGON_1):
                    if object_id not in counted_ids:
                        counted_ids[object_id] = True
                        # Trigger a one-frame blink.
                        self.blink_once = True
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

            # Use the new function to decide the polygon color (blink only one frame when triggered).
            polygon_color = self.get_polygon_color()
            cv2.polylines(frame, [POLYGON_1], isClosed=True, color=polygon_color, thickness=2)

            with frame_lock:
                current_frame = frame

            try:
                roi = hailo.get_roi_from_buffer(buffer)
                detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            except Exception as e:
                print(f"Hailo processing error: {str(e)}")
                return Gst.FlowReturn.ERROR

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
        finally:
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    def draw_detections(self, frame):
        """Draw bounding boxes and labels on the frame."""
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
        """Start the GStreamer pipeline."""
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        if hasattr(self, 'pipeline'):
            self.pipeline.set_state(Gst.State.NULL)
        if hasattr(self, 'loop'):
            self.loop.quit()

class StreamGenerator:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url

    def gstreamer_pipeline(self):
        return (
            f"rtspsrc location={self.rtsp_url} latency=50 drop-on-latency=true ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink sync=false"
        )

    def generate_frames(self):
        cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print("Error: Couldn't open RTSP stream. Check the URL, credentials, and network connectivity.")
            return
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to retrieve frame from stream. Retrying...")
                cap.release()
                cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
                if not cap.isOpened():
                    print("Error: Couldn't reopen RTSP stream.")
                    break
                continue
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()

def run_hls_pipeline():
    logging.basicConfig(filename='/home/pi/Documents/vehicleCounting/hls_pipeline.log', level=logging.DEBUG)
    logging.info("Starting HLS pipeline...")

    # Define the GStreamer pipeline string.
    pipeline_str = (
        "appsrc name=src is-live=true block=true format=time ! "
        "videoconvert ! video/x-raw,format=I420,width=1280,height=720 ! "
        "x264enc tune=zerolatency bitrate=1024 speed-preset=veryfast ! "
        "h264parse config-interval=1 ! "
        "mpegtsmux ! "
        "hlssink location=/home/pi/Documents/vehicleCounting/hls/segment%05d.ts "
        "playlist-location=/home/pi/Documents/vehicleCounting/hls/playlist.m3u8 "
        "target-duration=1 max-files=4"
    )
    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name("src")
    pipeline.set_state(Gst.State.PLAYING)
    
    # Set the caps for appsrc so that it knows our frame format.
    caps = Gst.Caps.from_string("video/x-raw,format=BGR,width=1280,height=720")
    appsrc.set_property("caps", caps)
    
    while True:
        # Get the most recent frame from your detection pipeline.
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()
        # Convert frame to raw bytes.
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts = int(time.time() * Gst.SECOND)
        buf.duration = Gst.SECOND // 30  # Assuming ~30 fps
        
        # Push the frame into the pipeline.
        ret = appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            logging.error(f"Error pushing buffer into HLS pipeline: {ret}")
        time.sleep(1/20)  # Sleep to maintain ~30 fps


    pipeline.set_state(Gst.State.NULL)
    logging.info("HLS pipeline stopped.")
    #If HLS not run, chmod -R 755 to the hls folder
    
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
