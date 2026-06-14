import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

def crawl_nyc_weather_2024(output_file='NYC_Weather_2024.csv'):
    # 1. Cài đặt session có cache và retry
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # 2. Định nghĩa tham số: Thêm cloudcover và windspeed_10m
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.7128,  # Tọa độ New York
        "longitude": -74.0060,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "hourly": ["temperature_2m", "precipitation", "cloudcover", "windspeed_10m"],
        "timezone": "America/New_York"
    }

    print("Đang tải dữ liệu từ Open-Meteo API...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    # 3. Lấy dải thời gian
    time_index = pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )

    # 4. Trích xuất mảng dữ liệu (Numpy arrays)
    hourly_data = {
        "time": time_index,
        "temperature_2m (°C)": hourly.Variables(0).ValuesAsNumpy(),
        "precipitation (mm)": hourly.Variables(1).ValuesAsNumpy(),
        "cloudcover (%)": hourly.Variables(2).ValuesAsNumpy(),
        "windspeed_10m (km/h)": hourly.Variables(3).ValuesAsNumpy()
    }

    # 5. Tạo DataFrame
    df = pd.DataFrame(data = hourly_data)

    # Lưu ý: Open-Meteo API trả về time timezone aware (UTC),
    # nhưng để dễ chạy với đoạn code hiện tại của bạn, ta có thể bỏ tz info
    # hoặc chuyển đổi thẳng về Local Time (nếu cần). 
    df['time'] = df['time'].dt.tz_localize(None)

    # 6. Lưu file CSV
    df.to_csv(output_file, index=False)
    print(f"Đã lưu file dữ liệu thô tại: {output_file}")
    
    return df

# Chạy thử
if __name__ == "__main__":
    raw_df_2024 = crawl_nyc_weather_2024('NYC_Weather_2024_Raw.csv')
    print(raw_df_2024.head())