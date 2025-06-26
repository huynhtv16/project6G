import os
import json
import csv
import time
import tracemalloc
import psutil
EARTH_RADIUS = 6371  # Bán kính Trái Đất tính bằng km
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# Config
REQUESTS_DIR = "dataset/requests"
LOG_SEC = "dataset/csv/log_seconds.csv"
LOG_MIN = "dataset/csv/log_minute.csv"


process = psutil.Process(os.getpid())

class RequestAnalyzer:
    def __init__(self):
        self.car_stats = {}
        self._init_csv_files()
        
    def _init_csv_files(self):
        # Khởi tạo file log
        for path, headers in [(LOG_SEC, ["timestamp","car_id","speed","distance","time","ram"]),
                             (LOG_MIN, ["timestamp","avg_speed","total_distance","avg_ram"])]:
            if not os.path.exists(path):
                with open(path, "w") as f:
                    csv.writer(f).writerow(headers)

    def _get_ram(self):
        return process.memory_info().rss / (1024 ** 2)  # MB

    def _haversine(self, loc1, loc2):
        # Chuyển đổi tọa độ sang radian
        lat1 = radians(loc1['latitude'])
        lon1 = radians(loc1['longitude'])
        lat2 = radians(loc2['latitude'])
        lon2 = radians(loc2['longitude'])
        
        # Tính chênh lệch tọa độ
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Áp dụng công thức Haversine
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return EARTH_RADIUS * c  # Khoảng cách tính bằng km

    def _process_file(self, filepath):
        tracemalloc.start()  # Bắt đầu theo dõi bộ nhớ

        try:
            print(f"\n Đang xử lý: {filepath}")
            with open(filepath) as f:
                data = json.load(f)
            
            distance = self._haversine(data["path"][0], data["path"][1])
            est_time = (distance / data["speed"]) * 60 if data["speed"] > 0 else 0

            current, peak = tracemalloc.get_traced_memory()
            ram_used = peak / (1024 ** 2)  # Đổi ra MB
            ip = data.get("ip","unknow")
            with open(LOG_SEC, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    data["car_id"],
                    data["speed"],
                    ip,
                    f"{distance:.2f}",
                    f"{est_time:.2f}",
                    f"{ram_used:.2f}"
                ])

        except Exception as e:
            print(f" Lỗi xử lý file {filepath}: {str(e)}")
            ram_used = 0
        
        finally:
            tracemalloc.stop()  # Ngừng theo dõi

    #

    def _log_minute_stats(self):
        try:
            print("\n Kiểm tra dữ liệu log phút (theo từng xe)...")
            with open(LOG_SEC, "r") as f:
                reader = csv.DictReader(f)
                car_data = {}

                for row in reader:
                    car_id = row.get("car_id")
                    if car_id and all(k in row for k in ['speed', 'distance', 'ram']):
                        try:
                            speed = float(row["speed"])
                            distance = float(row["distance"])
                            ram = float(row["ram"])
                            # Giả sử bạn lưu địa chỉ IP trong LOG_SEC trong tương lai
                            ip = row.get("ip", "unknown")
                        except ValueError:
                            continue

                        if car_id not in car_data:
                            car_data[car_id] = {"rows": [], "ip": ip}
                        
                        car_data[car_id]["rows"].append((speed, distance, ram))

            with open(LOG_MIN, "a", newline="") as f:
                writer = csv.writer(f)

                for car_id, data in car_data.items():
                    rows = data["rows"]
                    ip = data["ip"]
                    if not rows:
                        continue

                    count = len(rows)
                    avg_speed = sum(r[0] for r in rows) / count
                    total_distance = sum(r[1] for r in rows)
                    total_ram = sum(r[2] for r in rows)

                    writer.writerow([
                        datetime.now().isoformat(),
                        car_id,
                        round(avg_speed, 2),
                        round(total_distance, 2),
                        round(total_ram, 2),
                        count,
                        ip
                    ])
                    print(f" Ghi log phút cho xe {car_id}: avg_speed={avg_speed:.2f}, distance={total_distance:.2f}, total_ram={total_ram:.2f}, count={count}, ip={ip}")

        except Exception as e:
            print(f" Lỗi nghiêm trọng khi ghi log phút: {str(e)}")


    def run(self):
        processed = set()
        last_minute = time.time()
        
        while True:
            # Xử lý các file mới
            for fname in os.listdir(REQUESTS_DIR):
                if fname.endswith(".json") and fname not in processed:
                    try:
                        self._process_file(os.path.join(REQUESTS_DIR, fname))
                        processed.add(fname)
                    except Exception as e:
                        print(f"Lỗi xử lý {fname}: {str(e)}")

            # Ghi log phút
            if time.time() - last_minute > 60:
                self._log_minute_stats()
                last_minute = time.time()
            
            time.sleep(1)

if __name__ == "__main__":
    analyzer = RequestAnalyzer()
    analyzer.run()