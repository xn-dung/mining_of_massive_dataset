import os
import requests
import re
import urllib.request
from tqdm import tqdm

def download_yellow_taxi_data(years, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tlc_url = 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page'

    print(f"Đang lấy dữ liệu từ TLC cho các năm: {list(years)}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    response = requests.get(tlc_url, headers=headers)

    if response.status_code != 200:
        print(f"Không thể truy cập trang web TLC. Mã lỗi: {response.status_code}")
        return

    for target_year in years:
        print(f"\n--- BẮT ĐẦU TẢI NĂM {target_year} ---")

        pattern = rf'https://[^"]*yellow_tripdata_{target_year}-\d{{2}}\.parquet'
        parquet_links = list(set(re.findall(pattern, response.text)))

        if not parquet_links:
            print(f"Không tìm thấy dữ liệu Yellow Taxi cho năm {target_year}.")
            continue

        print(f"Tìm thấy {len(parquet_links)} files cho năm {target_year}.")

        for file_url in sorted(parquet_links):
            file_name = file_url.split('/')[-1]
            file_path = os.path.join(output_dir, file_name)

            if os.path.exists(file_path):
                print(f"File {file_name} đã tồn tại. Bỏ qua.")
                continue

            print(f'Đang tải {file_name}...')

            try:
                req_with_headers = urllib.request.Request(file_url, headers=headers)

                response_stream = urllib.request.urlopen(req_with_headers)
                file_size = int(response_stream.info().get('Content-Length', -1))

                with open(file_path, 'wb') as file, tqdm(
                    desc=file_name, total=file_size, unit='B', unit_scale=True, unit_divisor=1024
                ) as bar:
                    block_size = 8192
                    while True:
                        buffer = response_stream.read(block_size)
                        if not buffer:
                            break
                        file.write(buffer)
                        bar.update(len(buffer))

            except urllib.error.HTTPError as e:
                print(f"Lỗi HTTP {e.code} khi tải {file_name}: {e.reason}")
            except Exception as e:
                print(f"Lỗi không xác định khi tải {file_name}: {e}")

if __name__ == "__main__":
    TARGET_YEARS = range(2022, 2025)
    OUTPUT_DIR = "/content/drive/MyDrive/Mining of massive dataset/final_code/data/taxi_raw"
    download_yellow_taxi_data(TARGET_YEARS, OUTPUT_DIR)