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
