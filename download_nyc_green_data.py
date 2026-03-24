import os
import urllib.request
from tqdm import tqdm

YEAR = 2013 
MONTH = 8
END_YEAR = 2024
END_MONTH = 12
BASE_URL = 'https://d37ci6vzurychx.cloudfront.net'

while YEAR <= END_YEAR:
    year_dir = f'green_taxi_{YEAR}'
    os.makedirs(year_dir, exist_ok=True)

    while MONTH <= 12:
        if YEAR == END_YEAR and MONTH > END_MONTH:
            break

        month_str = str(MONTH).zfill(2)

        file_name = f'green_tripdata_{YEAR}-{month_str}.parquet'
        file_url = f'{BASE_URL}/trip-data/{file_name}'
        
        try:
            print(f'Downloading {file_name}...')

            response = urllib.request.urlopen(file_url)
            file_size = int(response.info().get('Content-Length', -1))

            save_path = os.path.join(year_dir, file_name)
            with open(save_path, 'wb') as file, \
                 tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=file_size, desc=file_name) as progress:
                block_size = 8192
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    file.write(buffer)
                    progress.update(len(buffer))
        
        except urllib.error.HTTPError as e:
            print(f'Lỗi: Không tìm thấy file {file_name} (HTTP {e.code}). Có thể tháng này chưa có dữ liệu.')

        MONTH += 1

    YEAR += 1
    MONTH = 1

print('Tất cả tệp dữ liệu Green Taxi đã được tải xong.')