import pandas as pd
import os
import requests
from sklearn.preprocessing import MinMaxScaler

def crawl_and_process_weather(years, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        print(f"\nĐang tải và xử lý dữ liệu thời tiết NYC năm {year}...")

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=40.7128&longitude=-74.0060&"
            f"start_date={year}-01-01&end_date={year}-12-31&"
            f"hourly=temperature_2m,precipitation,cloudcover,windspeed_10m&"
            f"timezone=America%2FNew_York"
        )

        response = requests.get(url).json()

        df = pd.DataFrame(response['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        target_columns = ['temperature_2m', 'precipitation', 'cloudcover', 'windspeed_10m']
        df[target_columns] = df[target_columns].apply(pd.to_numeric, errors='coerce')
        df = df.resample('30min').interpolate(method='linear')

        weather_df = df.copy()

        scaler = MinMaxScaler()
        continuous_cols = ['temperature_2m', 'cloudcover', 'windspeed_10m']
        weather_df[continuous_cols] = scaler.fit_transform(weather_df[continuous_cols])

        weather_df['Rain_None'] = (weather_df['precipitation'] == 0.0).astype(int)
        weather_df['Rain_Light'] = ((weather_df['precipitation'] > 0.0) & (weather_df['precipitation'] <= 2.5)).astype(int)
        weather_df['Rain_Moderate'] = ((weather_df['precipitation'] > 2.5) & (weather_df['precipitation'] <= 7.6)).astype(int)
        weather_df['Rain_Heavy'] = (weather_df['precipitation'] > 7.6).astype(int)
        weather_df.drop(columns=['precipitation'], inplace=True)

        output_path = os.path.join(output_dir, f'weather_{year}_processed.csv')
        weather_df.to_csv(output_path)
        print(f"-> Đã lưu: {output_path}")

if __name__ == "__main__":
    TARGET_YEARS = range(2018, 2025)
    OUTPUT_DIR = "/content/drive/MyDrive/Mining of massive dataset/final_code/DVC_storage/data/context"

    crawl_and_process_weather(TARGET_YEARS, OUTPUT_DIR)