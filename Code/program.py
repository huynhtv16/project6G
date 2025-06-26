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

import os
os.system(' del result\\attacker.log result\\server.log 2>nul')  # Xóa file log cũ nếu có

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
SERVER_THREAD_COUNT = 1000  # Số luồng xử lý cho server
sent_history = []
received_history = []
time_history = []
received_lock = threading.Lock()  # Lock để bảo vệ received_packets

def ddos_attack():
    global packet_count
    while True:
        packet = {"packet_id": str(uuid.uuid4()), "timestamp": int(time.time())}
        packet_queue.put(packet)
        packet_count += 1
        attacker_logger.info(f"Generated packet {packet['packet_id']}")
        time.sleep(0.001)  # Giảm sleep để gửi nhanh hơn

def server_listener():
    global received_packets
    while True:
        if not packet_queue.empty():
            packet = packet_queue.get()
            with received_lock:
                received_packets += 1
            server_logger.info(f"Received Packet {packet['packet_id']} at {packet['timestamp']} ")
        time.sleep(0.001)  # Giảm sleep để xử lý nhanh hơn

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