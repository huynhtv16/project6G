import os
import json
import uuid
import math
import random
from datetime import datetime
import time

# Config
REQUESTS_DIR = "dataset/requests"
CAR_CENTER = (100.0, 100.0)  # Lat, Lon
EARTH_RADIUS = 6371  # km

class RequestGenerator:
    def __init__(self):
        self.cars = {}  # Lưu trữ thông tin xe
        self.reuse_prob = 0.7  # Xác suất dùng lại xe cũ

    def _generate_location(self, center_lat, center_lon, radius_km):
        # Chuyển đổi bán kính sang radian
        radius_rad = radius_km / EARTH_RADIUS
        
        # Tạo góc phương vị ngẫu nhiên và khoảng cách
        bearing = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, radius_rad)
        
        # Chuyển đổi tọa độ trung tâm sang radian
        lat1 = math.radians(center_lat)
        lon1 = math.radians(center_lon)
        
        # Tính toán tọa độ mới
        lat2 = math.asin(
            math.sin(lat1) * math.cos(distance) +
            math.cos(lat1) * math.sin(distance) * math.cos(bearing)
        )
        
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(distance) * math.cos(lat1),
            math.cos(distance) - math.sin(lat1) * math.sin(lat2)
        )
        
        # Chuyển đổi về độ và trả về
        return {
            "latitude": math.degrees(lat2),
            "longitude": math.degrees(lon2)
        }
    def _create_new_car(self):
        car_id = str(uuid.uuid4())[:8].upper()
        return {
            "id": car_id,
            "ip": ".".join(str(random.randint(0, 255)) for _ in range(4)),
            "requests": 0
        }

    def generate(self):
        # Chọn xe mới hoặc cũ
        if self.cars and random.random() < self.reuse_prob:
            car_id = random.choice(list(self.cars.keys()))
            car = self.cars[car_id]
        else:
            car = self._create_new_car()
            self.cars[car["id"]] = car

        # Tạo request
        car["requests"] += 1
        start = self._generate_location(*CAR_CENTER, 2)
        end = self._generate_location(start["latitude"], start["longitude"], 100)

        return {
            "car_id": car["id"],
            "ip": car["ip"],
            "speed": random.randint(30, 120),
            "path": [start, end],
            "timestamp": datetime.now().isoformat()
        }

    def save_request(self):
        os.makedirs(REQUESTS_DIR, exist_ok=True)
        data = self.generate()
        filename = f"{data['car_id']}_{int(time.time())}.json"  # Đã sửa ở đây
        
        with open(f"{REQUESTS_DIR}/{filename}", "w") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    generator = RequestGenerator()
    while True:
        generator.save_request()
        time.sleep(random.uniform(0.1, 1))