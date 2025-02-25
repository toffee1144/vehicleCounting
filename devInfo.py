#!/usr/bin/env python3
import psutil
import time
import subprocess
from hailo_platform import Device
import socket
import platform
import subprocess
import json
import glob
import os

def __init__(update_interval=1):
    """
    Initialize the SystemMonitor.

    :param update_interval: Time in seconds between updates.
    """
    update_interval = update_interval

def get_uptime():
    """
    Calculate the system uptime.

    :return: A tuple containing (days, hours, minutes, seconds)
    """
    boot_time = psutil.boot_time()
    now = time.time()
    uptime_seconds = now - boot_time
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return {
        'days': int(days),
        'hours': int(hours),
        'minutes': int(minutes),
        'seconds': int(seconds)
    }

def get_memory_info():
    """
    Get system memory information.

    :return: A dictionary with keys 'total', 'available', 'used', 'percent'
    """
    mem = psutil.virtual_memory()
    return {
        'total': mem.total,
        'available': mem.available,
        'used': mem.used,
        'percent': mem.percent
    }

def get_disk_info():
    """
    Get detailed disk partition information.

    :return: A list of dictionaries, one per partition.
    """
    partitions = psutil.disk_partitions(all=False)
    disk_info = []
    for partition in partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            # Skip partitions where permission is denied.
            continue
        disk_info.append({
            'device': partition.device,
            'mountpoint': partition.mountpoint,
            'fstype': partition.fstype,
            'total': usage.total,
            'used': usage.used,
            'free': usage.free,
            'percent': usage.percent
        })
    return disk_info

def display():
    """Clear the screen and display uptime, memory, and disk information."""
    # Display uptime
    get_uptime()

    # Display memory information
    get_memory_info()

    # Display disk partitions information
    get_disk_info()

def get_cpu_temp():
    """
    Retrieves the CPU temperature.
    First, tries to use 'vcgencmd' (specific to Raspberry Pi).
    If that fails, it attempts to read from the thermal zone file.
    """
    try:
        # Using vcgencmd, which is common on Raspberry Pi.
        output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("UTF-8")
        # Output is expected in the format: temp=42.0'C
        temp_str = output.split("=")[1].split("'")[0]
        return float(temp_str)
    except Exception:
        # Fallback: Read from thermal_zone0 if available.
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_str = f.readline().strip()
                # The value is usually in millidegree Celsius.
                return float(temp_str) / 1000
        except Exception:
            return None
        
def get_hailo_temp():
    target = Device()
    temp = target.control.get_chip_temperature().ts0_temperature
    return round(float(temp), 2)

#####################

def get_ip_address():
    """Retrieve the local IP address by creating a dummy socket connection."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # The connection does not have to succeed – it only forces the system to assign an IP.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "Unable to determine IP"
    finally:
        s.close()
    return ip

def get_device_serial():
    """Extract the unique Raspberry Pi serial number from /proc/cpuinfo."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith("Serial"):
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return "N/A"

def get_mac_address(interface):
    """Retrieve the MAC address for a given network interface (e.g., eth0 or wlan0)."""
    try:
        path = f'/sys/class/net/{interface}/address'
        with open(path, 'r') as f:
            return f.readline().strip()
    except Exception:
        return "N/A"
    
def get_hailo_device_info():
    """
    Runs the command 'hailortcli fw-control identify' to obtain Hailo device details,
    parses the output, and returns a dictionary containing:
      - Board Name
      - Device Architecture
      - Serial Number
      - Part Number

    Returns:
        dict: A dictionary with device information or None if the command fails.
    """
    try:
        # Execute the command and capture the output.
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            check=True
        )
    except FileNotFoundError:
        print("hailortcli command not found. Ensure that the Hailo SDK is installed and in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e.stderr.strip()}")
        return None

    device_info = {}
    # Parse each line of the output.
    
    for line in result.stdout.splitlines():
        if ':' in line:
            key, value = line.split(":", 1)
            key = key.strip()
            # Remove null characters from the value (e.g., "\x00")
            value = value.strip().replace('\x00', '')
            device_info[key] = value
    return device_info

def get_speed_test():
    command = ["speedtest", "--format=json"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    download_bandwidth = data['download']['bandwidth'] * 8 / 1_000_000  # Convert to Mbps
    upload_bandwidth = data['upload']['bandwidth'] * 8 / 1_000_000  # Convert to Mbps

    summary = {
        'download_mbps': download_bandwidth,
        'upload_mbps': upload_bandwidth
    }

    return data, summary

def delete_hls_file():
    hls_dir = "./hls"
    
    # Create the directory if it doesn't exist.
    if not os.path.exists(hls_dir):
        os.makedirs(hls_dir)
    else:
        # Delete any old HLS segment files.
        for segment_file in glob.glob(os.path.join(hls_dir, "segment*.ts")):
            try:
                os.remove(segment_file)
                print(f"Deleted old segment: {segment_file}")
            except Exception as e:
                print(f"Error deleting {segment_file}: {e}")
                
        # Delete the playlist if it exists.
        playlist_file = os.path.join(hls_dir, "playlist.m3u8")
        if os.path.exists(playlist_file):
            try:
                os.remove(playlist_file)
                print("Deleted old playlist file.")
            except Exception as e:
                print(f"Error deleting {playlist_file}: {e}")