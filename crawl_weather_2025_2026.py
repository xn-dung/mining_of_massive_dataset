import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os
from sklearn.preprocessing import MinMaxScaler

def get_date_range(start, end):
    """Hàm hỗ trợ lấy danh sách chuỗi ngày giữa 2 mốc"""
    return pd.date_range(start=start, end=end).strftime('%Y-%m-%d').tolist()

def fetch_and_process_weather_holidays(output_dir):
    print("1. Đang tải dữ liệu thời tiết từ Open-Meteo API...")
    # --- PHẦN 1: CÀO DỮ LIỆU ---
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "start_date": "2025-01-01",
        "end_date": "2025-12-31", # Lấy dữ liệu đến hết tháng 4/2026
        "hourly": ["temperature_2m", "precipitation", "cloudcover", "windspeed_10m"],
        "timezone": "America/New_York"
    }

    responses = openmeteo.weather_api(url, params=params)
    hourly = responses[0].Hourly()

    time_index = pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )

    df = pd.DataFrame({
        "time": time_index,
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "precipitation": hourly.Variables(1).ValuesAsNumpy(),
        "cloudcover": hourly.Variables(2).ValuesAsNumpy(),
        "windspeed_10m": hourly.Variables(3).ValuesAsNumpy()
    })
    
    # Đưa về Local Time (bỏ Timezone)
    df['time'] = df['time'].dt.tz_localize(None)
    df.set_index('time', inplace=True)

    print("2. Đang xử lý dữ liệu: Resample 30 phút, Chuẩn hóa, One-Hot Encoder...")
    # --- PHẦN 2: TIỀN XỬ LÝ (RESAMPLE & SCALE) ---
    target_columns = ['temperature_2m', 'precipitation', 'cloudcover', 'windspeed_10m']
    df[target_columns] = df[target_columns].apply(pd.to_numeric, errors='coerce')

    # Resample 30 phút/lần
    df = df.resample('30min').interpolate(method='linear')
    weather_df = df.copy()

    # Chuẩn hóa Min-Max
    scaler = MinMaxScaler()
    continuous_cols = ['temperature_2m', 'cloudcover', 'windspeed_10m']
    weather_df[continuous_cols] = scaler.fit_transform(weather_df[continuous_cols])

    # One-hot lượng mưa
    weather_df['Rain_None'] = (weather_df['precipitation'] == 0.0).astype(int)
    weather_df['Rain_Light'] = ((weather_df['precipitation'] > 0.0) & (weather_df['precipitation'] <= 2.5)).astype(int)
    weather_df['Rain_Moderate'] = ((weather_df['precipitation'] > 2.5) & (weather_df['precipitation'] <= 7.6)).astype(int)
    weather_df['Rain_Heavy'] = (weather_df['precipitation'] > 7.6).astype(int)
    weather_df.drop(columns=['precipitation'], inplace=True)

    print("3. Đang ánh xạ lịch nghỉ lễ (Holidays) cho năm 2025 và T1-T4/2026...")
    # --- PHẦN 3: XỬ LÝ HOLIDAYS 2025-2026 ---
    holidays_df = pd.DataFrame(index=weather_df.index)
    date_strs = holidays_df.index.strftime('%Y-%m-%d')

    # 1. LEGAL HOLIDAYS
    legal_holidays = [
        # Năm 2025
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', '2025-06-19', '2025-07-04', 
        '2025-09-01', '2025-10-13', '2025-11-04', '2025-11-11', '2025-11-27', '2025-12-25',
    ]

    # 2. SCHOOL RECESS
    school_recess = (
        # Năm 2025
        ['2025-01-01', '2025-01-29', '2025-03-31', '2025-06-06', '2025-09-23', '2025-09-24', '2025-10-02', '2025-10-20', '2025-11-28'] +
        get_date_range('2025-02-17', '2025-02-21') + # Midwinter Recess 2025
        get_date_range('2025-04-14', '2025-04-18') + # Spring Recess 2025
        get_date_range('2025-12-24', '2025-12-31')   # Winter Recess 2025
    )

    # 3. EVENT FESTIVALS
    event_festivals = [
        # Năm 2025
        '2025-02-14', '2025-03-17', '2025-04-20', '2025-05-11', '2025-06-15', '2025-10-31', '2025-12-24', '2025-12-31'
    ]

    # 4. EARLY DISMISSAL 
    early_dismissal = [
        '2025-03-06', '2025-03-13', '2025-05-08', '2025-05-15', '2025-11-06', '2025-11-13'
    ]

    holidays_df['Is_Legal_Holiday'] = date_strs.isin(legal_holidays).astype(int)
    holidays_df['Is_School_Recess'] = date_strs.isin(school_recess).astype(int)
    holidays_df['Is_Event_Festival'] = date_strs.isin(event_festivals).astype(int)
    holidays_df['Is_Early_Dismissal'] = date_strs.isin(early_dismissal).astype(int)

    # --- PHẦN 4: LƯU FILE ---
    os.makedirs(output_dir, exist_ok=True)
    weather_path = os.path.join(output_dir, 'weather_2025.csv')
    holidays_path = os.path.join(output_dir, 'holidays_2025.csv')

    weather_df.to_csv(weather_path)
    holidays_df.to_csv(holidays_path)

    print("\nHOÀN TẤT! Đã tạo và lưu thành công 2 file tại:")
    print(f"- {weather_path}")
    print(f"- {holidays_path}")

if __name__ == "__main__":
    output_dir = r'D:\StudyStuff\Mining Massive Dataset\mining_of_massive_dataset'
    fetch_and_process_weather_holidays(output_dir)