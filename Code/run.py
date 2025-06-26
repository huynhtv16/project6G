import subprocess
import time
import os
import sys

def main():
    # Đảm bảo path hợp lệ dù chạy ở đâu
    attacker_script = os.path.join("code", "attacker.py")
    server_script = os.path.join("code", "fake_server.py")

    # Mở hai chương trình trong hai cửa sổ terminal riêng biệt
    if sys.platform == "win32":
        # Windows: mở trong 2 cửa sổ cmd riêng
        server_process = subprocess.Popen(["start", "cmd", "/k", f"python {server_script}"], shell=True)
        attacker_process = subprocess.Popen(["start", "cmd", "/k", f"python {attacker_script}"], shell=True)

    else:
        # Mac / Linux: mở terminal mới
        server_process = subprocess.Popen(["gnome-terminal", "--", "python3", server_script])
        attacker_process = subprocess.Popen(["gnome-terminal", "--", "python3", attacker_script])

    print("🛡️ Fake Server & 💣 Attacker đã được khởi động.")
    print("Nhấn Ctrl + C để dừng toàn bộ!")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⛔ Kết thúc phiên mô phỏng. Dừng tiến trình server & attacker.")
        # Trên Windows: process là cửa sổ CMD, nên tự đóng sẽ kết thúc.
        if sys.platform != "win32":
            attacker_process.terminate()
            server_process.terminate()

        print("✅ Đã dừng attacker và server.")

if __name__ == "__main__":
    main()
