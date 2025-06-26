import csv
import re
import os

# Mở file SQL
sql_folder = r'D:\DH\K2_N3\HQT_CSDL_ORACLE\CSDL'
with open(os.path.join(sql_folder, 'DBqlyQuanToc.sql'), 'r', encoding='utf-8') as file:
    content = file.read()

# Tìm các dòng INSERT INTO
inserts = re.findall(r"INSERT INTO `?(\w+)`?.*?VALUES\s*(.*?);", content, re.DOTALL)

for table, values_str in inserts:
    # Bỏ dấu ngoặc ngoài
    values = re.findall(r"\((.*?)\)", values_str)

    # Ghi ra CSV trong thư mục CSDL
    csv_path = os.path.join(sql_folder, f"{table}.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in values:
            fields = [x.strip(" '\"") for x in row.split(',')]
            writer.writerow(fields)

print("Đã chuyển xong sang CSV")