import threading
import time
import uuid
import logging
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
import psutil
import json
import os
import random
import math

# Xóa file log cũ nếu có
os.system('del result\\attacker.log result\\server.log 2>nul')

# Tạo logger riêng cho attacker
attacker_logger = logging.getLogger('attacker')
attacker_logger.setLevel(logging.INFO)
attacker_handler = logging.FileHandler('result\\attacker.log')
attacker_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
attacker_logger.addHandler(attacker_handler)

# Tạo logger riêng cho server
server_logger = logging.getLogger('server')
server_logger.setLevel(logging.INFO)
server_handler = logging.FileHandler('result\\server.log')
server_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
server_logger.addHandler(server_handler)

console = Console()
packet_queue = multiprocessing.Queue()

# Biến toàn cục
packet_count = 0
received_packets = 0
THREAD_COUNT = 1000
SERVER_THREAD_COUNT = 1000
sent_history = []
received_history = []
time_history = []
received_lock = threading.Lock()

# Tích hợp PacketGenerator từ create_packet.py
CAR_CENTER = (100.0, 100.0)  # Lat, Lon
EARTH_RADIUS = 6371  # km

class PacketGenerator:
    def __init__(self, center_lat=CAR_CENTER[0], center_lon=CAR_CENTER[1], radius_km=1):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_km = radius_km
        self.packet_id = 0
        self.attack_types = ["SYN_Flood", "UDP_Flood", "HTTP_Flood"]
        self.destination_ips = ["10.0.0.50", "172.16.0.10", "192.168.1.200"]
        self.destination_ports = [80, 443, 53, 22]

    def _generate_location(self):
        radius_rad = self.radius_km / EARTH_RADIUS
        bearing = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, radius_rad)

        lat1 = math.radians(self.center_lat)
        lon1 = math.radians(self.center_lon)

        lat2 = math.asin(
            math.sin(lat1) * math.cos(distance) +
            math.cos(lat1) * math.sin(distance) * math.cos(bearing)
        )

        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(distance) * math.cos(lat1),
            math.cos(distance) - math.sin(lat1) * math.sin(lat2)
        )

        return {
            "latitude": math.degrees(lat2),
            "longitude": math.degrees(lon2)
        }

    def _generate_ip(self):
        return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"

    def _generate_payload(self, attack_type):
        if attack_type == "HTTP_Flood":
            return f"GET / HTTP/1.1\r\nHost: example.com\r\nUser-Agent: Mozilla/5.0\r\n\r\n"
        elif attack_type == "UDP_Flood":
            return ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=100))
        else:
            return ""

    def _create_new_packet(self):
        self.packet_id += 1
        location = self._generate_location()
        attack_type = random.choice(self.attack_types)
        protocol = "TCP" if attack_type in ["SYN_Flood", "HTTP_Flood"] else "UDP"
        packet_type = "SYN" if attack_type == "SYN_Flood" else "GET" if attack_type == "HTTP_Flood" else "DATAGRAM"
        flags = ["SYN"] if attack_type == "SYN_Flood" else []

        packet = {
            "packet_id": str(self.packet_id).zfill(8),
            "timestamp": int(time.time()),
            "location": location,
            "source_ip": self._generate_ip(),
            "destination_ip": random.choice(self.destination_ips),
            "source_port": random.randint(1024, 65535),
            "destination_port": random.choice(self.destination_ports),
            "protocol": protocol,
            "packet_type": packet_type,
            "payload": self._generate_payload(attack_type),
            "ttl": random.randint(32, 255),
            "flags": flags,
            "attack_type": attack_type,
            "volume": random.randint(64, 1500),
            "spoofed": random.choice([True, False])
        }
        return packet

# Tạo instance của PacketGenerator
generator = PacketGenerator(radius_km=20)

def ddos_attack():
    global packet_count
    while True:
        packet = generator._create_new_packet()
        packet_queue.put(packet)
        packet_count += 1
        attacker_logger.info(f"Generated packet {packet['packet_id']}, Attack Type: {packet['attack_type']}")
        time.sleep(0.001)

def server_listener():
    global received_packets
    while True:
        if not packet_queue.empty():
            packet = packet_queue.get()
            with received_lock:
                received_packets += 1
            # Đánh giá RAM và CPU khi nhận gói tin
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            server_logger.info(
                f"Received Packet {packet['packet_id']} at {packet['timestamp']}, "
                f"Attack Type: {packet['attack_type']}, Volume: {packet['volume']} bytes, "
                f"CPU: {cpu}%, RAM: {ram}%"
            )
        time.sleep(0.001)

def display_status():
    start_time = time.time()
    while time.time() - start_time < 60:  # Chạy 1 phút
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        
        # Lưu lịch sử để vẽ biểu đồ
        elapsed_time = int(time.time() - start_time)
        time_history.append(elapsed_time)
        sent_history.append(packet_count)
        received_history.append(received_packets)
        
        # Hiển thị bảng
        table = Table(title="🌐 DDoS Simulation Status")
        table.add_column("Threads Running (Attack)", justify="center")
        table.add_column("Threads Running (Server)", justify="center")
        table.add_column("Total Packets Sent", justify="center")
        table.add_column("Total Packets Received", justify="center")
        table.add_column("CPU Usage (%)", justify="center")
        table.add_column("RAM Usage (%)", justify="center")
        table.add_row(
            str(THREAD_COUNT),
            str(SERVER_THREAD_COUNT),
            str(packet_count),
            str(received_packets),
            f"{cpu}%",
            f"{ram}%"
        )
        console.clear()
        console.print(table)
        time.sleep(1)
    
    # Vẽ biểu đồ sau khi chạy xong
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, sent_history, label='Packets Sent', color='blue')
    plt.plot(time_history, received_history, label='Packets Received', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Packets')
    plt.title('Packets Sent and Received Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('packets_over_time.png')

def main():
    # Khởi động các thread cho attacker
    for _ in range(THREAD_COUNT):
        threading.Thread(target=ddos_attack, daemon=True).start()
    
    # Khởi động nhiều thread cho server
    for _ in range(SERVER_THREAD_COUNT):
        threading.Thread(target=server_listener, daemon=True).start()

    print("🛡️ Fake Server & 💣 Attacker đã khởi động. Chạy trong 1 phút...")
    display_status()

    print("\n⛔ Kết thúc phiên mô phỏng.")
    print("✅ Đã dừng simulation. Biểu đồ lưu tại 'packets_over_time.png'.")
    exit()

if __name__ == "__main__":
    main()