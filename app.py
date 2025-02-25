from flask import Flask, Response, jsonify, render_template, request, url_for, send_from_directory
import cv2
import devInfo
import platform
import datetime
import uuid
import platform
from database import get_rtsp_link_from_db, get_data_all, get_data_summary, get_playback_rtsp_link
import subprocess
import threading
import time
from detection import StreamGenerator

app = Flask(__name__, static_folder='hls')
appData = Flask(__name__)

playback_links = {}

@app.route('/device/api/v1/stream/live')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Serve playlist and segments from the same directory.
@app.route('/hls/<path:filename>')
def hls_files(filename):
    return send_from_directory(app.static_folder, filename)

@appData.route('/device/api/v1/disk', methods=['GET'])
def get_disk_info():
    disk_data = {
        'Uptime': devInfo.get_uptime(),          # Call the function
        'Memory': devInfo.get_memory_info(),       # Call the function
        'Disk': devInfo.get_disk_info(),            # Call the function
        'CPU Temp': devInfo.get_cpu_temp(),         # Call the function
        'Hailo Temp': devInfo.get_hailo_temp()     # Call the function
    }
    return jsonify(disk_data)

@appData.route('/device/api/v1/info', methods=['GET'])
def get_raspi_info():
    raspi_info = {
        'Hostname': platform.node(),
        'Operating System': f"{platform.system()} {platform.release()}",
        'IP Address': devInfo.get_ip_address(),
        'MAC Address Ethernet': devInfo.get_mac_address('eth0'),
        'MAC Address WiFi': devInfo.get_mac_address('eth0'),
        'Device Serial': devInfo.get_device_serial(),
        'Hailo Device Info': devInfo.get_hailo_device_info(),
    }
    return jsonify(raspi_info)

@appData.route('/device/api/v1/speedtest', methods=['GET'])
def get_speedtest_info():
    speedtest_info = {
        'Speedtest': devInfo.get_speed_test()
    }
    return jsonify(speedtest_info)

@appData.route('/device/api/v1/data/all', methods=['GET'])
def get_data_all_info():
    return jsonify(get_data_all())

@appData.route('/device/api/v1/data/summary', methods=['GET'])
def get_data_summary_info():
    return jsonify(get_data_summary())

@appData.route('/device/api/v1/stream/playback', methods=['GET', 'POST'])
def playback():
    if request.method == 'POST':
        datetime_str = request.form['datetime']
        playback_id = str(uuid.uuid4())
        expires_at = datetime.datetime.now() + datetime.timedelta(minutes=3)
        playback_links[playback_id] = {
            'datetime': datetime_str,
            'expires_at': expires_at
        }
        return jsonify({
            'playback_link': url_for('playback_stream', playback_id=playback_id, _external=True),
            'expires_at': expires_at.isoformat()
        })
    return '''
        <form method="post">
            DateTime: <input type="datetime-local" name="datetime">
            <input type="submit" value="Generate">
        </form>
    '''

@appData.route('/device/api/v1/stream/playback/<playback_id>', methods=['GET'])
def playback_stream(playback_id):
    if playback_id not in playback_links:
        return "Playback link not found", 404
    playback_info = playback_links[playback_id]
    if datetime.datetime.now() > playback_info['expires_at']:
        del playback_links[playback_id]
        return "Playback link expired", 410
    datetime_str = playback_info['datetime']
    base_rtsp_url = get_rtsp_link_from_db()
    
    rtsp_url = get_playback_rtsp_link(base_rtsp_url, datetime_str)
    print("Playback RTSP URL:", rtsp_url)
    stream_generator = StreamGenerator(rtsp_url)
    return Response(stream_generator.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@appData.route('/device/api/v1/stream/playback/list', methods=['GET'])
def list_playback_links():
    playback_list = [
        {'playback_id': playback_id, 'datetime': info['datetime']}
        for playback_id, info in playback_links.items()
    ]
    return jsonify(playback_list)

@appData.route('/device/api/v1/service/restart')
def restart_service():
    def delayed_reboot():
        time.sleep(3)
        subprocess.run(["sudo", "reboot"])

    # Start the delayed reboot in a separate thread
    threading.Thread(target=delayed_reboot).start()
    return jsonify({'result': 'Device will reset in 3 seconds'})
