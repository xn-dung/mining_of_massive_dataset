import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def process_and_export_features_2024(file_path, output_dir):
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

    # Lọc lấy năm 2024
    df_2024 = df.loc['2024-01-01':'2024-12-31'].copy()

    # Ép kiểu dữ liệu về dạng số
    target_columns = ['temperature_2m', 'precipitation', 'cloudcover', 'windspeed_10m']
    df_2024[target_columns] = df_2024[target_columns].apply(pd.to_numeric, errors='coerce')

    # Resample 30 phút/lần và nội suy tuyến tính
    df_2024 = df_2024.resample('30min').interpolate(method='linear')
    df_base = df_2024[target_columns].copy()

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

    def get_date_range(start, end, year='2024'):
        return pd.date_range(start=f'{year}-{start}', end=f'{year}-{end}').strftime('%Y-%m-%d').tolist()

    # 1. LEGAL HOLIDAYS 2024 (Đã kiểm tra chuẩn năm 2024)
    # 01-01: New Year, 01-15: MLK Day (Thứ 2 tuần 3), 02-19: Presidents Day (Thứ 2 tuần 3), 
    # 05-27: Memorial Day, 06-19: Juneteenth, 07-04: Independence Day, 09-02: Labor Day, 
    # 10-14: Columbus Day, 11-05: Election Day, 11-11: Veterans Day, 11-28: Thanksgiving, 12-25: Christmas
    legal_holidays = [f'2024-{d}' for d in ['01-01', '01-15', '02-19', '05-27', '06-19', '07-04', '09-02', '10-14', '11-05', '11-11', '11-28', '12-25']]
    
    # 2. SCHOOL RECESS 2024 (Theo lịch NYC DOE 2023-2024 & 2024-2025)
    # Bao gồm nghỉ đông, nghỉ xuân, và các ngày lễ tôn giáo mà trường học nghỉ
    school_recess = (
        get_date_range('02-19', '02-23') + # Midwinter Recess
        get_date_range('04-22', '04-30') + # Spring Recess (Kéo dài do Lễ Vượt Qua)
        get_date_range('12-24', '12-31') + # Winter Recess
        [f'2024-{d}' for d in ['03-29', '04-01', '04-10', '06-06', '10-03', '10-04', '11-01', '11-29']] 
        # (Lễ Phục Sinh, Eid al-Fitr, Anniversary Day, Rosh Hashanah, Diwali, Black Friday)
    )
    
    # 3. EVENT FESTIVALS 2024 (Các ngày lễ hội, mua sắm)
    # 03-31: Easter Sunday (Năm 2024 là 31/3), 05-12: Mother's Day, 06-16: Father's Day
    event_festivals = [f'2024-{d}' for d in ['02-14', '03-17', '03-31', '05-12', '06-16', '10-31', '12-24', '12-31']]
    
    # 4. EARLY DISMISSAL 2024 (Mô phỏng các ngày họp phụ huynh ở NYC)
    # Thường rơi vào Thứ Năm của tháng 3, 5, và 11.
    early_dismissal = [f'2024-{d}' for d in ['03-07', '03-14', '05-09', '05-16', '11-07', '11-14']]

    date_strs = holidays_df.index.strftime('%Y-%m-%d')

    holidays_df['Is_Legal_Holiday'] = date_strs.isin(legal_holidays).astype(int)
    holidays_df['Is_School_Recess'] = date_strs.isin(school_recess).astype(int)
    holidays_df['Is_Event_Festival'] = date_strs.isin(event_festivals).astype(int)
    holidays_df['Is_Early_Dismissal'] = date_strs.isin(early_dismissal).astype(int)

    # Tạo thư mục nếu nó chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Khai báo đường dẫn file hoàn chỉnh
    weather_path = os.path.join(output_dir, 'weather_2024.csv')
    holidays_path = os.path.join(output_dir, 'holidays_2024.csv')

    # Lưu file
    weather_df.to_csv(weather_path)
    holidays_df.to_csv(holidays_path)

    return weather_df, holidays_df, weather_path, holidays_path

if __name__ == "__main__":
    file_path = "D:/StudyStuff/Mining Massive Dataset/mining_of_massive_dataset/NYC_Weather_2024_Raw.csv"
    output_dir = "D:/StudyStuff/Mining Massive Dataset/mining_of_massive_dataset"

    try:
        w_df, h_df, w_path, h_path = process_and_export_features_2024(file_path, output_dir)

        print("Đã tạo và lưu thành công 2 file tại:")
        print(f"1. {w_path}")
        print(f"2. {h_path}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file gốc '{file_path}'. Vui lòng kiểm tra lại đường dẫn.")