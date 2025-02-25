import time
import datetime
import pytz
import mysql.connector
import paho.mqtt.client as mqtt
import threading
import json
from decimal import Decimal

# Set timezone
jakarta_tz = pytz.timezone("Asia/Jakarta")

# Global offline data buffer in case the database connection fails.
offline_data = []

prev_counts = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0,
}
def flush_offline_data(cursor, conn):
    """Attempt to flush any stored offline data to the database."""
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

def handle_database_error(car, motorcycle, bus, truck, current_timestamp):
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

def mysql_insertion_loop(app_instance):
    """Continuously insert count data into the MySQL database every 5 seconds."""
    while True:
        time.sleep(60)
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
                last_date = result[1]  # Assuming DATE(time) is in the second column.
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
                        INSERT INTO counting (car, motorcycle, bus, truck, time)
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

def get_rtsp_link_from_db():
    """Retrieve the RTSP link from the database."""
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
    
def get_playback_rtsp_link(url: str, datetime_str: str):
    if not url:
        return "RTSP link not found in the database", 404
    
    jakarta_tz = pytz.timezone("Asia/Jakarta")
    gmt_tz = pytz.timezone("GMT")

    dt_jakarta = jakarta_tz.localize(datetime.datetime.fromisoformat(datetime_str))
    dt_gmt = dt_jakarta.astimezone(gmt_tz)
    
    # Convert datetime to the required format
    formatted_datetime = dt_gmt.strftime('%Y%m%dT%H%M%SZ')
    
    # Construct the RTSP URL dynamically
    return f"{url.replace('/Channels/', '/tracks/')}/?starttime={formatted_datetime}"

def get_data_all():
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
    
## 

def get_data_summary():
    # Establish the database connection.
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="telkomiot123",
        database="AI_Vehicle"
    )
    cursor = conn.cursor(dictionary=True)
    
    # Get the 'date' query parameter if provided.
    
    # Prepare the SQL query. If a date is provided, use it; otherwise, use CURDATE().
    query = """
        SELECT 
            SUM(car) AS total_car,
            SUM(bus) AS total_bus,
            SUM(motorcycle) AS total_motorcycle,
            SUM(truck) AS total_truck
        FROM counting
        WHERE DATE(time) = CURDATE()
    """
    cursor.execute(query)
    
    result = cursor.fetchone()
    
    # Close the database connection.
    cursor.close()
    conn.close()
    return result

def get_latest_data():
    """Retrieve the total all-time and today's data from the database."""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="telkomiot123",
            database="AI_Vehicle"
        )
        cursor = conn.cursor(dictionary=True)
        
        # Query for all-time totals
        cursor.execute("""
            SELECT 
                CAST(SUM(car) AS UNSIGNED) AS total_car,
                CAST(SUM(bus) AS UNSIGNED) AS total_bus,
                CAST(SUM(motorcycle) AS UNSIGNED) AS total_motorcycle,
                CAST(SUM(truck) AS UNSIGNED) AS total_truck
            FROM counting
        """)
        total_all = cursor.fetchone()
        
        # Query for today's totals (using CURDATE() to match records from today)
        cursor.execute("""
            SELECT 
                CAST(SUM(car) AS UNSIGNED) AS total_car,
                CAST(SUM(bus) AS UNSIGNED) AS total_bus,
                CAST(SUM(motorcycle) AS UNSIGNED) AS total_motorcycle,
                CAST(SUM(truck) AS UNSIGNED) AS total_truck
            FROM counting
            WHERE DATE(time) = CURDATE()
        """)
        total_today = cursor.fetchone()

        cursor.execute("""
            SELECT 
                serialNumber AS sn,
                name as deviceName
            FROM deviceInfo
        """)
        device_info = cursor.fetchone()

        cursor.close()
        conn.close()
        
        # Ensure that if any of the values are None (no records), they default to 0.
        data = {
            "sn": device_info["sn"],  # SN from the database.
            "name": device_info["deviceName"],
            "total_all": {
                "car": total_all["total_car"] if total_all["total_car"] is not None else 0,
                "bus": total_all["total_bus"] if total_all["total_bus"] is not None else 0,
                "truck": total_all["total_truck"] if total_all["total_truck"] is not None else 0,
                "motor": total_all["total_motorcycle"] if total_all["total_motorcycle"] is not None else 0
            },
            "total_today": {
                "car": total_today["total_car"] if total_today["total_car"] is not None else 0,
                "bus": total_today["total_bus"] if total_today["total_bus"] is not None else 0,
                "truck": total_today["total_truck"] if total_today["total_truck"] is not None else 0,
                "motor": total_today["total_motorcycle"] if total_today["total_motorcycle"] is not None else 0
            }
        }
        return data
    except Exception as e:
        print("Error retrieving data from database:", e)
        return None

def send_data_via_mqtt(app_instance):
    """Send data via MQTT at regular intervals using difference of cumulative counts."""
    client = mqtt.Client()
    client.connect("36.92.168.180", 7483, 60)
    
    def publish_data():
        data = get_latest_data()
        if data:
            # Calculate the difference between current counts and previous counts.
            delta_car = app_instance.car_crossed_count - prev_counts["car"]
            delta_motorcycle = app_instance.motorcycle_crossed_count - prev_counts["motorcycle"]
            delta_bus = app_instance.bus_crossed_count - prev_counts["bus"]
            delta_truck = app_instance.truck_crossed_count - prev_counts["truck"]
            
            # Include the difference in the payload.
            data["total_5_second"] = {
                "car": delta_car,
                "motorcycle": delta_motorcycle,
                "bus": delta_bus,
                "truck": delta_truck,
                "timestamp": datetime.datetime.now(jakarta_tz).isoformat()
            }
            # Format the topic using the SN from the data.
            topic = f"device/{data['sn']}/vehicle"
            
            # Update previous counts for the next iteration.
            prev_counts["car"] = app_instance.car_crossed_count
            prev_counts["motorcycle"] = app_instance.motorcycle_crossed_count
            prev_counts["bus"] = app_instance.bus_crossed_count
            prev_counts["truck"] = app_instance.truck_crossed_count
        else:
            topic = "device/unknown/vehicle"
        
        client.publish(topic, json.dumps(data))
        print("Data sent via MQTT on topic", topic, ":", data)
        threading.Timer(5.0, publish_data).start()
    
    publish_data()
