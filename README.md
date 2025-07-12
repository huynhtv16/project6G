# 🛡️ Mô phỏng và Phát hiện Tấn công DDoS bằng Deep Learning

Dự án mô phỏng các cuộc tấn công từ chối dịch vụ phân tán (DDoS) và huấn luyện mô hình học sâu (CNN, RNN) để phân loại các loại tấn công dựa trên log máy chủ. Phù hợp với nghiên cứu bảo mật mạng và học máy.

## 🚀 Chức năng chính

- 💣 Mô phỏng tấn công DDoS theo thời gian thực với 3 kiểu: **SYN Flood**, **UDP Flood**, **HTTP Flood**
- 🌐 Máy chủ giả lập xử lý gói tin, ghi log chi tiết từng packet nhận được
- 📊 Ghi nhận và hiển thị thời gian thực: số packet gửi/nhận, băng thông, CPU, RAM
- 🧠 Huấn luyện mô hình **CNN** và **RNN** từ dữ liệu log để phát hiện loại tấn công
- 📈 Tạo báo cáo đánh giá mô hình (Accuracy, F1-score, Confusion Matrix)
- 📷 Lưu biểu đồ trực quan về mạng và hệ thống

## 🧪 Công nghệ sử dụng

- **Ngôn ngữ:** Python
- **Thư viện học máy:** PyTorch, scikit-learn
- **Xử lý dữ liệu:** Pandas, Regex, ipaddress
- **Trực quan hóa:** matplotlib, seaborn, rich
- **Hệ thống & log:** multiprocessing, threading, logging, signal

## 📂 Cấu trúc thư mục
```plaintext
📦 DDoS-Detection-Simulator
│
├── code/ # Tự động tạo ra khi chạy mô phỏng
│ ├── DDoS_attacker.py # bắt đầu tấn công gửi các file log đến server
│ └── CNN_RNN.py # Bắt đầu xử lý dữ liệu từ file log huấn luyện model 
│
├── images/ # Thư mục chứa các biểu đồ đã vẽ
│ ├── CNN_confusion_matrix.png
│ ├── RNN_confusion_matrix.png
│ └── pps_cpu_ram_over_time.png (tự sinh)
│
├── Results/ # Đánh giá mô hình
│ ├── CNN.txt # Báo cáo mô hình CNN
│ └── RNN.txt # Báo cáo mô hình RNN
│
├── packets_over_time.png # Biểu đồ số lượng packet gửi / nhận
├── cnn_ddos_classifier.pth # Trọng số mô hình CNN đã huấn luyện
├── rnn_ddos_classifier.pth # Trọng số mô hình RNN đã huấn luyện
│
├── simulate_ddos.py # File mô phỏng tấn công và log hệ thống
├── train_and_detect.py # File huấn luyện và đánh giá mô hình
└── README.md # Mô tả dự án (file này)
```

## 📸 Hình ảnh minh họa
<img width="1256" height="750" alt="image" src="https://github.com/user-attachments/assets/84251fed-6018-4a12-82d2-c9ecde9f221e" />
<img width="880" height="528" alt="image" src="https://github.com/user-attachments/assets/4bfbc647-4898-43e5-889f-8aac5b6d74df" />
<img width="913" height="753" alt="image" src="https://github.com/user-attachments/assets/01df3022-d9a5-4940-a666-7262be850d27" />


📚 Ghi chú
Dự án không sử dụng cơ sở dữ liệu mà phân tích trực tiếp từ file log (server.log)

Có thể mở rộng lưu log vào SQLite hoặc MongoDB nếu cần

Phù hợp làm đề tài học thuật hoặc nghiên cứu về phát hiện tấn công mạng
