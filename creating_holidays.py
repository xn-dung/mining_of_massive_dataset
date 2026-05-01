import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def process_and_export_features(file_path, output_dir):
    df = pd.read_csv(file_path)

    # Đổi tên cột (loại bỏ phần đơn vị nếu có) để tránh lỗi KeyError
    df.rename(columns={
        'temperature_2m (°C)': 'temperature_2m',
        'precipitation (mm)': 'precipitation',
        'cloudcover (%)': 'cloudcover',
        'windspeed_10m (km/h)': 'windspeed_10m'
    }, inplace=True)

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Lọc lấy năm 2018
    df_2018 = df.loc['2018-01-01':'2018-12-31'].copy()

    # Ép kiểu dữ liệu về dạng số để resample và interpolate không bị lỗi
    target_columns = ['temperature_2m', 'precipitation', 'cloudcover', 'windspeed_10m']
    df_2018[target_columns] = df_2018[target_columns].apply(pd.to_numeric, errors='coerce')

    # Resample 30 phút/lần và nội suy tuyến tính
    df_2018 = df_2018.resample('30min').interpolate(method='linear')
    df_base = df_2018[target_columns].copy()

    weather_df = df_base.copy()

    # Chuẩn hóa Min-Max
    scaler = MinMaxScaler()
    continuous_cols = ['temperature_2m', 'cloudcover', 'windspeed_10m']
    weather_df[continuous_cols] = scaler.fit_transform(weather_df[continuous_cols])

    # One-hot precipitation
    weather_df['Rain_None'] = (weather_df['precipitation'] == 0.0).astype(int)
    weather_df['Rain_Light'] = ((weather_df['precipitation'] > 0.0) & (weather_df['precipitation'] <= 2.5)).astype(int)
    weather_df['Rain_Moderate'] = ((weather_df['precipitation'] > 2.5) & (weather_df['precipitation'] <= 7.6)).astype(int)
    weather_df['Rain_Heavy'] = (weather_df['precipitation'] > 7.6).astype(int)

    # Xóa cột gốc
    weather_df.drop(columns=['precipitation'], inplace=True)

    holidays_df = pd.DataFrame(index=df_base.index)

    def get_date_range(start, end, year='2018'):
        return pd.date_range(start=f'{year}-{start}', end=f'{year}-{end}').strftime('%Y-%m-%d').tolist()

    legal_holidays = [f'2018-{d}' for d in ['01-01', '01-15', '02-12', '02-19', '05-28', '07-04', '09-03', '10-08', '11-06', '11-12', '11-22', '12-25']]
    school_recess = (
        get_date_range('01-02', '01-05') +
        [f'2018-{d}' for d in ['01-26', '01-29']] +
        get_date_range('02-16', '02-23') +
        get_date_range('03-30', '04-06') +
        [f'2018-{d}' for d in ['06-07', '06-11', '06-15']]
    )
    event_festivals = [f'2018-{d}' for d in ['02-14', '03-17', '04-01', '05-13', '06-17', '10-31', '12-24']]
    early_dismissal = [f'2018-{d}' for d in ['03-06', '03-09', '03-13', '03-15', '06-26']]

    date_strs = holidays_df.index.strftime('%Y-%m-%d')

    holidays_df['Is_Legal_Holiday'] = date_strs.isin(legal_holidays).astype(int)
    holidays_df['Is_School_Recess'] = date_strs.isin(school_recess).astype(int)
    holidays_df['Is_Event_Festival'] = date_strs.isin(event_festivals).astype(int)
    holidays_df['Is_Early_Dismissal'] = date_strs.isin(early_dismissal).astype(int)

    # Tạo thư mục nếu nó chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Khai báo đường dẫn file hoàn chỉnh
    weather_path = os.path.join(output_dir, 'weather_2018.csv')
    holidays_path = os.path.join(output_dir, 'holidays_2018.csv')

    # Lưu file
    weather_df.to_csv(weather_path)
    holidays_df.to_csv(holidays_path)

    return weather_df, holidays_df, weather_path, holidays_path

if __name__ == "__main__":
    file_path = '/content/drive/MyDrive/Mining of massive dataset/preprocessing_output/NYC_Weather_2016_2022.csv'
    output_dir = '/content/drive/MyDrive/Mining of massive dataset/preprocessing_output'

    try:
        w_df, h_df, w_path, h_path = process_and_export_features(file_path, output_dir)

        print("Đã tạo và lưu thành công 2 file tại:")
        print(f"1. {w_path}")
        print(f"2. {h_path}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file gốc '{file_path}'. Vui lòng kiểm tra lại đường dẫn.")